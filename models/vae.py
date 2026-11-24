# -*- coding: utf-8 -*-
"""视觉自编码器兼容层（模块解耦后）。

编码器 / 解码器已拆分为独立模块：
  - models/encoder.py  : ConvEncoder
  - models/decoder.py  : ConvDecoder

本文件保留两个类的转发导入（向后兼容所有历史脚本与权重），并保留 ConvVAE 封装。
历史 checkpoint 的权重 key（encoder.* / decoder.*）与拆分前完全一致。
"""
from typing import Dict

import torch
import torch.nn as nn

from models.encoder import ConvEncoder
from models.decoder import ConvDecoder

__all__ = ["ConvEncoder", "ConvDecoder", "ConvVAE"]


class ConvVAE(nn.Module):
    """编码器 + 解码器封装：支持重参数化采样，或直接取 mu（纯 AE 模式）。"""

    def __init__(self, latent_dim: int = 32, hidden_channels: int = 32,
                 in_channels: int = 3, out_size: int = 64):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.out_size = int(out_size)
        self.encoder = ConvEncoder(in_channels, self.latent_dim, hidden_channels)
        self.decoder = ConvDecoder(self.latent_dim, in_channels, hidden_channels, out_size)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, sample: bool = False) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon = self.decoder(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ConvVAE(latent_dim=32).eval()
    out = model(torch.randn(2, 3, 64, 64))
    assert out["recon"].shape == (2, 3, 64, 64)
    print("VAE_SHIM_OK", model.count_parameters())
