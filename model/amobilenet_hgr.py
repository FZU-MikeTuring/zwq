"""
A-MobileNet-HGR: 面向智能家居手势识别的改进 MobileNet 模型

架构: MobileNetV3-Small（保留原生 SE）→ ECA 通道注意力
      → SoftSpatial 空间注意力 → GAP → 分类头
"""
import torch
import torch.nn as nn
import torchvision

from .eca import ECALayer
from .spatial_attention import SoftSpatialAttention


class AMobileNetGesture(nn.Module):
    """A-MobileNet-HGR: MobileNetV3-Small + ECA + SoftSpatial"""

    def __init__(self, num_classes: int = 7, dropout1: float = 0.25,
                 dropout2: float = 0.20, hidden_dim: int = 128):
        super().__init__()

        # 骨干（保留原生 SE，ImageNet 预训练权重完整）
        base = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features = base.features  # (B, 576, 7, 7)

        # 注意力（骨干出口，不破坏预训练特征）
        self.eca = ECALayer(k_size=3)
        self.spatial = SoftSpatialAttention(kernel_size=7)

        # 分类头
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout1),
            nn.Linear(576, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(p=dropout2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)      # 骨干（含原生 SE）
        x = self.eca(x)           # ECA 通道注意力
        x = self.spatial(x)       # SoftSpatial 空间注意力
        x = self.pool(x)          # GAP
        x = self.classifier(x)
        return x

    @property
    def param_count(self) -> float:
        """参数量（百万）"""
        return sum(p.numel() for p in self.parameters()) / 1e6
