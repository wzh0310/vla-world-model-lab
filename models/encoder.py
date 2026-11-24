# -*- coding: utf-8 -*-
"""视觉编码器模块（模块解耦后独立）。

ConvEncoder: 4 层卷积逐级下采样 + 自适应池化到 4x4 + GroupNorm，
输出观测分布参数 mu / logvar。adaptive pool 使网络可接受任意分辨率输入
（分辨率需为 16 的倍数）；输入 64x64 时卷积输出恰为 4x4，池化为恒等，
与历史 checkpoint 的权重 key / shape 完全一致。

该模块从原 models/vae.py 拆出。原 vae.py 保留 ConvEncoder 的转发导入，
因此所有历史脚本与权重不受影响。
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """卷积编码器：任意分辨率(16 的倍数) -> 隐分布 (mu, logvar)。"""

    def __init__(self, in_channels: int = 3, latent_dim: int = 32,
                 hidden_channels: int = 32):
        super().__init__()
        if latent_dim < 1 or hidden_channels < 1:
            raise ValueError("latent_dim 和 hidden_channels 必须为正整数")
        c1, c2, c3, c4 = (hidden_channels, hidden_channels * 2,
                          hidden_channels * 4, hidden_channels * 8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, 4, stride=2, padding=1), nn.GroupNorm(1, c1), nn.SiLU(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.GroupNorm(1, c2), nn.SiLU(),
            nn.Conv2d(c2, c3, 4, stride=2, padding=1), nn.GroupNorm(1, c3), nn.SiLU(),
            nn.Conv2d(c3, c4, 4, stride=2, padding=1), nn.GroupNorm(1, c4), nn.SiLU(),
        )  # 输出 (B, c4, H/16, W/16)
        self.flat_dim = c4 * 4 * 4
        self.head = nn.Linear(self.flat_dim, 2 * latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"编码器需要 (B,C,H,W) 图像，得到 {tuple(x.shape)}")
        if x.shape[-2] < 16 or x.shape[-1] < 16 or x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError(f"图像分辨率需为 16 的倍数且 >=16，得到 {tuple(x.shape[-2:])}")
        h = F.adaptive_avg_pool2d(self.conv(x), (4, 4)).flatten(1)
        mu, logvar = self.head(h).chunk(2, dim=1)
        return mu, logvar


if __name__ == "__main__":
    torch.manual_seed(0)
    enc = ConvEncoder(latent_dim=64).eval()
    mu, lv = enc(torch.randn(2, 3, 64, 64))
    assert mu.shape == (2, 64) and lv.shape == (2, 64)
    print("ENCODER_OK", tuple(mu.shape))
