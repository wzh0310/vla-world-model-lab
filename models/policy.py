# -*- coding: utf-8 -*-
"""极简 MLP 策略网络：在世界模型想象训练 Demo 中使用。

设计要点：
- 输入：RSSM 拼接隐状态特征 (B, det+stoch)，默认 264 维。
- 输出：4 维归一化动作，tanh 限制到 [-1,1]（与环境的归一化动作空间一致：
        [dx,dy,dz,gripper] 已由 fast_loader 归一到 [-1,1]）。
- 网络：256→128→4 两层 MLP + SiLU，输出层 tanh。
- 与 DreamerV2 的 actor 同构（确定性均值策略），但不含 log_std / 高斯采样，
  保持"最小可用"定位；动作随机性由世界模型 RSSM 的随机态重参数化提供。
"""
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """隐状态特征 -> 归一化动作 [-1,1]^act_dim 的确定性策略。"""

    def __init__(self, feat_dim: int = 264, act_dim: int = 4,
                 hidden: tuple = (256, 128)):
        super().__init__()
        h1, h2 = int(hidden[0]), int(hidden[1])
        self.net = nn.Sequential(
            nn.Linear(feat_dim, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, act_dim), nn.Tanh(),
        )
        self.feat_dim, self.act_dim = int(feat_dim), int(act_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat (B,F) -> action (B,A)，逐元素 ∈ [-1,1]。"""
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"特征维度应为 {self.feat_dim}，得到 {feat.shape[-1]}")
        return self.net(feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    p = MLPPolicy(264)
    feat = torch.randn(5, 264)
    a = p(feat)
    assert a.shape == (5, 4) and a.abs().max() <= 1.0
    print(f"POLICY_OK  输出={tuple(a.shape)} 范围=[{a.min():.3f},{a.max():.3f}] "
          f"参数量={p.count_parameters():,}")
