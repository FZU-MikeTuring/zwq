"""
全局配置：手势映射、模型参数、预处理参数、系统参数
对应论文中的表1和训练参数表
"""
import numpy as np

# ==================== 手势标签映射 ====================
# 论文表1：手势与智能家居控制指令映射
GESTURE_LABELS = {
    0: "palm",       # 手掌 → 开/关灯
    1: "fist",       # 握拳 → 关闭/停止
    2: "ok",         # OK → 确认/切换模式
    3: "like",       # 拇指向上（赞）→ 增加
    4: "dislike",    # 拇指向下（踩）→ 减少
    5: "one",        # 食指 → 切换设备
    6: "no_gesture", # 无手势 → 空闲（安全占位）
}

GESTURE_LABELS_CN = {
    0: "手掌",
    1: "握拳",
    2: "OK",
    3: "拇指向上",
    4: "拇指向下",
    5: "食指",
    6: "无手势",
}

# ==================== 控制指令映射 ====================
COMMAND_MAP = {
    "palm":       "light_on",
    "fist":       "light_off",
    "ok":         "confirm_mode",
    "like":       "increase",
    "dislike":    "decrease",
    "one":        "switch_device",
    "no_gesture": "idle",
}

COMMAND_DESC_CN = {
    "light_on":      "开灯",
    "light_off":     "关灯",
    "confirm_mode":  "确认模式",
    "increase":       "调高",
    "decrease":       "调低",
    "switch_device":  "切换设备",
    "idle":           "空闲",
}

NUM_CLASSES = 7

# ==================== 模型参数 ====================
# 论文表5：训练超参数
INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 训练超参数
BATCH_SIZE = 32
NUM_EPOCHS = 50
FREEZE_EPOCHS = 5          # 前5轮冻结骨干
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# ==================== 时域投票参数 ====================
SLIDING_WINDOW = 8          # 滑窗大小
MIN_VOTES = 5              # 最少一致票数
CONFIDENCE_THRESHOLD = 0.75 # 置信度阈值
COOLDOWN_SEC = 1.0         # 冷却时间（秒）

# ==================== 预处理参数 ====================
ROI_EXPAND_SCALE = 1.25    # ROI扩展比例
GAUSSIAN_BLUR_KSIZE = (3, 3)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# ==================== 模拟设备状态 ====================
DEFAULT_DEVICE_STATES = {
    "客厅灯": False,
    "空调": False,
    "窗帘": False,
    "电视": False,
}
