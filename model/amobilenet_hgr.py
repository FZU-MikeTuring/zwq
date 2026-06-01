"""
A-MobileNet-HGR: 基于 MobileNetV3-Small 的手势识别模型

架构: MobileNetV3-Small（ImageNet 预训练）→ GAP → 分类头
"""
import torch
import torch.nn as nn
import torchvision


class AMobileNetGesture(nn.Module):
    """MobileNetV3-Small + 分类头（ImageNet 预训练权重）"""

    def __init__(self, num_classes: int = 7, dropout1: float = 0.25,
                 dropout2: float = 0.20, hidden_dim: int = 128):
        super().__init__()

        # 骨干（ImageNet 预训练权重）
        base = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features = base.features  # 输出 (B, 576, 7, 7)

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
        x = self.features(x)      # 骨干
        x = self.pool(x)          # GAP
        x = self.classifier(x)
        return x

    @property
    def param_count(self) -> float:
        """参数量（百万）"""
        return sum(p.numel() for p in self.parameters()) / 1e6
