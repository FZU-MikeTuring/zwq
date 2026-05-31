"""
A-MobileNet-HGR: 面向智能家居手势识别的改进 MobileNet 模型
论文表3中的 AMobileNetGesture 完整实现

架构: MobileNetV3-Small 骨干 → ECA 通道注意力 →
      SoftSpatial 空间注意力 → 全局平均池化 → 分类头
"""
import torch
import torch.nn as nn
import torchvision

from .eca import ECALayer
from .spatial_attention import SoftSpatialAttention


class AMobileNetGesture(nn.Module):
    """
    A-MobileNet-HGR (A-MobileNet for Home Gesture Recognition)

    结构：
    1. MobileNetV3-Small features（冻结低层，微调高层）
    2. ECA 通道注意力 — 增强关键通道响应
    3. SoftSpatial 空间注意力 — 聚焦手部关键区域
    4. AdaptiveAvgPool2d → Flatten → Dropout → Linear+Hardswish → Dropout → Linear
    """

    def __init__(self, num_classes: int = 7, dropout1: float = 0.25,
                 dropout2: float = 0.20, hidden_dim: int = 128):
        """
        Args:
            num_classes: 手势类别数（默认7）
            dropout1: 第一个 Dropout 比率
            dropout2: 第二个 Dropout 比率
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # ---- 骨干网络 ----
        base = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features = base.features  # 输出通道数: 576

        # ---- 注意力模块 ----
        self.eca = ECALayer(k_size=3)
        self.spatial = SoftSpatialAttention(kernel_size=7)

        # ---- 分类头 ----
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
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            (B, num_classes) — 各类别 logits
        """
        x = self.features(x)
        x = self.eca(x)
        x = self.spatial(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def get_attention_maps(self, x: torch.Tensor):
        """
        返回中间特征用于可视化

        Args:
            x: (B, 3, 224, 224)
        Returns:
            features: 骨干网络输出的特征图 (B, 576, H', W')
            attn_eca: ECA 加权后的特征图
            attn_spatial: 空间注意力加权后的特征图
            logits: 分类 logits
        """
        features = self.features(x)            # 骨干输出
        attn_eca = self.eca(features)          # ECA 后
        attn_spatial = self.spatial(attn_eca)  # 空间注意力后
        pooled = self.pool(attn_spatial)
        logits = self.classifier(pooled)
        return features, attn_eca, attn_spatial, logits

    def load_pretrained(self, path: str, device: str = "cpu"):
        """加载训练好的权重"""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        print(f"✓ 已加载预训练权重: {path}")

    @property
    def param_count(self) -> float:
        """返回参数量（百万）"""
        return sum(p.numel() for p in self.parameters()) / 1e6
