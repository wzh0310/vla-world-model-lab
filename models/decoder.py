# -*- coding: utf-8 -*-
"""图像解码器模块（模块解耦后独立）。

ConvDecoder: 隐特征 z (feat_dim) -> 4x4 -> 4 层转置卷积 -> out_size x out_size x 3，
tanh 输出 [-1,1]。out_size=64 时 resize 为恒等，与历史权重完全兼容。

该模块从原 models/vae.py 拆出；原 vae.py 保留 ConvDecoder 的转发导入。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDecoder(nn.Module):
    """转置卷积解码器：隐变量 z -> out_size x out_size x 3（tanh 输出，[-1,1]）。"""

    def __init__(self, latent_dim: int = 32, out_channels: int = 3,
                 hidden_channels: int = 32, out_size: int = 64):
        super().__init__()
        c1, c2, c3, c4 = (hidden_channels, hidden_channels * 2,
                          hidden_channels * 4, hidden_channels * 8)
        self.flat_dim = c4 * 4 * 4
        self.out_size = int(out_size)
        self.project = nn.Linear(latent_dim, self.flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1), nn.GroupNorm(1, c3), nn.SiLU(),
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1), nn.GroupNorm(1, c2), nn.SiLU(),
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1), nn.GroupNorm(1, c1), nn.SiLU(),
            nn.ConvTranspose2d(c1, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )  # 固定输出 64x64

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"解码器只接受 (B,latent_dim) 隐变量，得到 {tuple(z.shape)}")
        h = self.project(z).view(-1, self.flat_dim // 16, 4, 4)
        img = self.deconv(h)
        if self.out_size != 64:
            img = F.interpolate(img, size=(self.out_size, self.out_size),
                                mode="bilinear", align_corners=False)
        return img


if __name__ == "__main__":
    torch.manual_seed(0)
    dec = ConvDecoder(latent_dim=264, out_size=64).eval()
    img = dec(torch.randn(2, 264))
    assert img.shape == (2, 3, 64, 64)
    print("DECODER_OK", tuple(img.shape))
