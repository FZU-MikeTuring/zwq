"""
SoftSpatial 空间注意力模块
论文表2中的 SoftSpatialAttention 实现

通过平均池化和最大池化沿通道维拼接，经单个卷积生成空间注意力图。
"""
import torch
import torch.nn as nn


class SoftSpatialAttention(nn.Module):
    """
    空间注意力模块
    沿通道维度取 avg 和 max，拼接后通过一个大核卷积生成空间权重
    """
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: 空间注意力的卷积核大小
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) — 空间加权后的特征图
        """
        # 沿通道维度取平均
        avg = torch.mean(x, dim=1, keepdim=True)       # (B, 1, H, W)
        # 沿通道维度取最大值
        mx, _ = torch.max(x, dim=1, keepdim=True)      # (B, 1, H, W)
        # 拼接后卷积
        weight = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B, 1, H, W)
        return x * weight
