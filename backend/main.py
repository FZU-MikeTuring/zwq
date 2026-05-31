"""
FastAPI 后端服务
提供 WebSocket 实时手势识别 + REST API + 前端静态文件服务
"""
import sys
import os
import time
import json
import asyncio
from io import BytesIO
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import AMobileNetGesture, GestureCommandFilter
from preprocessing import preprocess_frame, preprocess_frame_simple, expand_to_square
from config import (
    GESTURE_LABELS, GESTURE_LABELS_CN, COMMAND_MAP, COMMAND_DESC_CN,
    NUM_CLASSES, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    SLIDING_WINDOW, MIN_VOTES, CONFIDENCE_THRESHOLD, COOLDOWN_SEC,
    DEFAULT_DEVICE_STATES, ROI_EXPAND_SCALE,
    GAUSSIAN_BLUR_KSIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
)

# ==================== 应用初始化 ====================
app = FastAPI(
    title="A-MobileNet-HGR API",
    description="Gesture Recognition Smart Home Control System",
    version="1.0.0",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
device_states = DEFAULT_DEVICE_STATES.copy()
command_history = deque(maxlen=20)
hand_detector = None


def init_model(model_path: str = None):
    """初始化 A-MobileNet-HGR 模型"""
    global model
    model = AMobileNetGesture(num_classes=NUM_CLASSES).to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[OK] Model weights loaded: {model_path}")
    else:
        print("[WARN] Using ImageNet pretrained weights (not fine-tuned)")
    model.eval()
    return model


def init_hand_detector():
    """初始化手部检测器"""
    global hand_detector
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions

        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            print("[INFO] Downloading MediaPipe hand model...")
            urllib.request.urlretrieve(url, model_path)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_detector = vision.HandLandmarker.create_from_options(options)
        print("[OK] MediaPipe hand detector ready")
        return "mediapipe_new"
    except Exception:
        print("[INFO] Using skin color detection fallback")
        hand_detector = "skincolor"
        return "skincolor"


def detect_hand(frame_rgb):
    """检测手部边界框"""
    global hand_detector
    h, w = frame_rgb.shape[:2]

    if hand_detector == "mediapipe_new":
        try:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = hand_detector.detect(mp_image)
            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                return (max(0, int(min(xs))), max(0, int(min(ys))),
                        min(w, int(max(xs))), min(h, int(max(ys))))
        except Exception:
            pass

    # 肤色检测回退
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)

    lower_hsv1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_hsv2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, lower_hsv1, upper_hsv1),
        cv2.inRange(hsv, lower_hsv2, upper_hsv2),
    )

    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 3000:
            x, y, bw, bh = cv2.boundingRect(largest)
            return (x, y, x + bw, y + bh)
    return None


def process_frame(frame_bgr: np.ndarray, gesture_filter: GestureCommandFilter):
    """
    处理单帧：手部检测 → 预处理 → 推理 → 投票

    Returns:
        dict with prediction results, or None if no hand detected
    """
    global model, device

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    # 手部检测
    hand_box = detect_hand(frame_rgb)
    if hand_box is None:
        return None

    # 预处理
    try:
        input_tensor = preprocess_frame(frame_bgr, hand_box).to(device)
    except Exception:
        return None

    # 模型推理
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(1)
        confidence = confidence.item()
        pred_label = GESTURE_LABELS[pred_idx.item()]

    # 时域投票
    now = time.time()
    command = gesture_filter.update(pred_label, confidence, now)
    vote_status = gesture_filter.get_vote_status()

    # 更新设备状态
    if command and command != "idle":
        _update_device_state(command)
        command_history.append({
            "time": time.strftime("%H:%M:%S"),
            "gesture": GESTURE_LABELS_CN[pred_idx.item()],
            "command": COMMAND_DESC_CN.get(command, command),
            "confidence": confidence,
        })

    # ROI 预处理后图像（用于前端展示）
    roi_img = preprocess_frame_simple(frame_bgr, hand_box)
    _, roi_jpg = cv2.imencode(".jpg", cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))

    return {
        "type": "prediction",
        "pred_label": pred_label,
        "pred_label_cn": GESTURE_LABELS_CN[pred_idx.item()],
        "confidence": round(confidence, 4),
        "probabilities": [round(p, 4) for p in probs.cpu().numpy()[0].tolist()],
        "hand_box": list(hand_box),
        "roi_image": roi_jpg.tobytes().hex(),  # base16 encoded JPEG
        "vote_status": vote_status,
        "command": command,
        "command_desc": COMMAND_DESC_CN.get(command, "") if command else "",
        "device_states": device_states,
    }


def _update_device_state(command):
    """更新模拟设备状态"""
    global device_states
    if command == "light_on":
        device_states["客厅灯"] = True
    elif command == "light_off":
        device_states["客厅灯"] = False
    elif command == "confirm_mode":
        device_states["电视"] = not device_states["电视"]
    elif command == "increase":
        device_states["空调"] = True
    elif command == "decrease":
        device_states["空调"] = False
    elif command == "switch_device":
        device_states["窗帘"] = not device_states["窗帘"]


# ==================== 生命周期 ====================
@app.on_event("startup")
async def startup():
    init_model()
    init_hand_detector()
    print("[OK] A-MobileNet-HGR Backend Ready")
    print(f"     Device: {device}")
    print(f"     Model params: {model.param_count:.2f}M")


# ==================== WebSocket 实时推理 ====================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    gesture_filter = GestureCommandFilter(
        window=SLIDING_WINDOW, min_votes=MIN_VOTES,
        threshold=CONFIDENCE_THRESHOLD, cooldown=COOLDOWN_SEC,
    )

    try:
        while True:
            # 接收二进制帧
            data = await ws.receive_bytes()

            # 解码 JPEG
            nparr = np.frombuffer(data, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            # 处理帧
            result = process_frame(frame_bgr, gesture_filter)
            if result is None:
                result = {
                    "type": "no_hand",
                    "device_states": device_states,
                }

            # 发送结果（JSON）
            await ws.send_json(result)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await ws.close()
        except Exception:
            pass


# ==================== REST API ====================
@app.get("/api/status")
async def get_status():
    """获取当前系统状态"""
    return {
        "device_states": device_states,
        "command_history": list(command_history),
        "model_info": {
            "device": device,
            "params_m": round(model.param_count, 2) if model else 0,
        },
    }


@app.post("/api/reset")
async def reset_system():
    """重置系统状态"""
    global device_states, command_history
    device_states = DEFAULT_DEVICE_STATES.copy()
    command_history.clear()
    return {"status": "ok", "message": "System reset"}


@app.get("/api/config")
async def get_config():
    """获取手势标签和控制映射配置"""
    return {
        "gesture_labels": GESTURE_LABELS,
        "gesture_labels_cn": GESTURE_LABELS_CN,
        "command_map": COMMAND_MAP,
        "command_desc_cn": COMMAND_DESC_CN,
        "num_classes": NUM_CLASSES,
        "params": {
            "sliding_window": SLIDING_WINDOW,
            "min_votes": MIN_VOTES,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "cooldown_sec": COOLDOWN_SEC,
        },
    }


@app.post("/api/predict/image")
async def predict_image(file: bytes):
    """
    单张图片推理（非流式）
    用于测试或上传图片识别
    """
    nparr = np.frombuffer(file, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(400, "Invalid image")

    temp_filter = GestureCommandFilter(window=1, min_votes=0, threshold=0, cooldown=0)

    result = process_frame(frame_bgr, temp_filter)
    if result is None:
        return {"type": "no_hand", "message": "No hand detected"}

    return result


# ==================== 前端静态文件 ====================
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# 挂载静态资源（JS, CSS 等）
if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
