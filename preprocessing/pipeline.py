"""
预处理管线: ROI 裁剪 → 方形扩展 → 高斯模糊 → CLAHE → 缩放 → 标准化
"""
import cv2
import numpy as np
import torch

from config import (
    INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    ROI_EXPAND_SCALE, GAUSSIAN_BLUR_KSIZE,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
)


def expand_to_square(x1: int, y1: int, x2: int, y2: int,
                     scale: float = ROI_EXPAND_SCALE,
                     frame_w: int = 640, frame_h: int = 480) -> tuple:
    """将矩形框扩展为正方形并放大"""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    half = side / 2
    return (
        max(0, int(cx - half)),
        max(0, int(cy - half)),
        min(frame_w, int(cx + half)),
        min(frame_h, int(cy + half)),
    )


def _enhance_roi(crop: np.ndarray) -> np.ndarray:
    """BGR→RGB → 高斯模糊 → CLAHE(LAB) → RGB（共享增强流程）"""
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.GaussianBlur(crop, GAUSSIAN_BLUR_KSIZE, 0)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_frame(frame: np.ndarray, hand_box: tuple) -> torch.Tensor:
    """完整预处理，返回归一化 tensor (1, 3, 224, 224)"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_to_square(*hand_box, frame_w=w, frame_h=h)

    crop = frame[y1:y2, x1:x2]
    crop = _enhance_roi(crop)
    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))

    tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)  # (1, 3, 224, 224)


def preprocess_frame_simple(frame: np.ndarray, hand_box: tuple) -> np.ndarray:
    """简化预处理，返回 uint8 numpy 数组（用于前端可视化）"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_to_square(*hand_box, frame_w=w, frame_h=h)

    crop = frame[y1:y2, x1:x2]
    crop = _enhance_roi(crop)
    return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
