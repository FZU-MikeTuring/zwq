"""
ECA (Efficient Channel Attention) 通道注意力模块
论文表2中的 ECALayer 实现

通过一维卷积捕获局部跨通道交互，避免全连接降维带来的信息损失。
"""
import torch
import torch.nn as nn


class ECALayer(nn.Module):
    """
    ECA 通道注意力
    使用 1D 卷积在通道维度上建模局部依赖关系
    自适应卷积核大小: k = |(log2(C) + 1) / 2|_odd
    """
    def __init__(self, channels: int = None, k_size: int = 3):
        """
        Args:
            channels: 输入通道数（用于自适应计算 k_size）
            k_size: 卷积核大小，若提供 channels 则自动计算
        """
        super().__init__()
        if channels is not None:
            # 自适应计算卷积核大小
            k_size = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + 1) / 2))
            if k_size % 2 == 0:
                k_size += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) — 通道加权后的特征图
        """
        # y: (B, C, 1, 1)
        y = self.avg_pool(x)
        # 变形为 Conv1d 输入: (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        # 恢复形状: (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
