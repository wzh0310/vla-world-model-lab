# -*- coding: utf-8 -*-
"""视觉自编码器：ConvEncoder + ConvDecoder（DreamerV2 视觉通道的独立版本）。

- ConvEncoder: 4 层卷积 ( stride=2 逐级下采样 16 倍 ) + 自适应池化到 4x4 + 归一化，
  输出隐分布参数 mu / logvar。adaptive pool 使网络可接受任意分辨率输入
  （分辨率需为 16 的倍数）；当输入为 64x64 时卷积输出恰为 4x4，池化为恒等，
  与历史 checkpoint 完全兼容。
- ConvDecoder: 4 层转置卷积 (4->8->16->32->64) + 归一化，输出再 resize 到目标
  分辨率 out_size（默认 64）。out_size=64 时 resize 恒等，兼容历史权重。

归一化说明：使用 GroupNorm(num_groups=1)，即对所有通道做归一化，
是卷积特征图上「层归一化」的标准实现（对小 batch 稳定，不依赖 batch 统计）。

分辨率配置：由 configs/default.yaml 的 data.image_size 统一控制，经训练/评估脚本
  的 --image-size 传入。修改分辨率后需重新训练 AE 与 WM（数据口径随之改变）。
"""
from typing import Dict, Optional, Tuple

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
        # 自适应池化到 4x4：64 输入时卷积输出已为 4x4，此步恒等（兼容旧权重）
        h = F.adaptive_avg_pool2d(self.conv(x), (4, 4)).flatten(1)
        mu, logvar = self.head(h).chunk(2, dim=1)
        return mu, logvar


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
        h = self.project(z).view(-1, self.flat_dim // 16, 4, 4)  # c4 通道
        img = self.deconv(h)                                     # (B,C,64,64)
        if self.out_size != 64:
            img = F.interpolate(img, size=(self.out_size, self.out_size),
                                mode="bilinear", align_corners=False)
        return img


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
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        out_s = model(x, sample=True)
    print(f"输入图像: {tuple(x.shape)}")
    for name, tensor in out.items():
        print(f"  {name}: {tuple(tensor.shape)}")
    assert out["recon"].shape == x.shape, "重建形状与输入不匹配"
    assert out["mu"].shape == (2, 32) and out["logvar"].shape == (2, 32)
    assert out_s["z"].shape == (2, 32)
    assert not torch.equal(out_s["z"], out["z"]), "采样模式应产生随机隐变量"
    # 动态分辨率检查：按 out_size 构造，32 / 96 均应正常
    for S in (32, 96):
        m2 = ConvVAE(latent_dim=32, out_size=S).eval()
        xx = torch.randn(2, 3, S, S)
        with torch.no_grad():
            rr = m2(xx)["recon"]
        assert rr.shape == (2, 3, S, S), f"分辨率 {S} 重建形状异常 {tuple(rr.shape)}"
        print(f"分辨率 {S}: recon {tuple(rr.shape)} OK")
    print(f"SHAPE_OK  参数量={model.count_parameters():,}  重建范围=[{float(out['recon'].min()):.3f}, {float(out['recon'].max()):.3f}]")
