"""
预处理管线
论文表1中的 preprocess_frame 完整实现

流程：ROI裁剪 → 方形扩展 → BGR→RGB → 高斯模糊 → CLAHE(LAB) →
     尺寸归一化(224×224) → ImageNet标准化
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
    """
    将矩形框扩展为正方形，并按比例放大

    Args:
        x1, y1, x2, y2: 边界框坐标
        scale: 扩展比例
        frame_w, frame_h: 帧尺寸

    Returns:
        (nx1, ny1, nx2, ny2) 扩展且裁剪到帧边界内的正方形坐标
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    half = side / 2
    nx1 = max(0, int(cx - half))
    ny1 = max(0, int(cy - half))
    nx2 = min(frame_w, int(cx + half))
    ny2 = min(frame_h, int(cy + half))
    return nx1, ny1, nx2, ny2


def preprocess_frame(frame: np.ndarray, hand_box: tuple) -> torch.Tensor:
    """
    完整的预处理管线（对应论文表1的伪代码）

    Args:
        frame: 原始BGR帧 (H, W, 3)
        hand_box: (x1, y1, x2, y2) 手部边界框

    Returns:
        tensor: (1, 3, 224, 224) 已归一化的张量
    """
    h, w = frame.shape[:2]

    # 1. 扩展ROI为正方形
    x1, y1, x2, y2 = expand_to_square(
        *hand_box, scale=ROI_EXPAND_SCALE, frame_w=w, frame_h=h
    )

    # 2. 裁剪ROI
    crop = frame[y1:y2, x1:x2]

    # 3. BGR → RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # 4. 高斯模糊去噪
    crop = cv2.GaussianBlur(crop, GAUSSIAN_BLUR_KSIZE, 0)

    # 5. LAB色彩空间 + CLAHE 自适应直方图均衡
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=CLAHE_TILE_SIZE)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 6. 缩放到 224×224
    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))

    # 7. 转为 tensor 并归一化 (HWC → CHW, uint8 → float32)
    tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0

    # 8. ImageNet 标准化
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)  # 增加 batch 维度: (1, 3, 224, 224)


def preprocess_frame_simple(frame: np.ndarray, hand_box: tuple) -> np.ndarray:
    """
    简化版预处理（返回 numpy 数组，便于可视化）
    不做 ImageNet 标准化，保持 uint8 格式
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_to_square(
        *hand_box, scale=ROI_EXPAND_SCALE, frame_w=w, frame_h=h
    )
    crop = frame[y1:y2, x1:x2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.GaussianBlur(crop, GAUSSIAN_BLUR_KSIZE, 0)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=CLAHE_TILE_SIZE)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
    return crop
