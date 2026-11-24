# -*- coding: utf-8 -*-
"""隐空间随机策略网络（Actor-Critic）—— 想象策略训练的基础组件。

与旧版 models/policy.py（确定性均值策略，仅用于最小 Demo）不同，本模块提供
DreamerV2 风格的高斯随机 Actor 与状态价值 Critic，供"世界模型内训练策略"使用：

  Actor  : 输入 RSSM 拼接隐特征 (B, F)，输出 4 维动作高斯分布的均值 mu 与
           对数标准差 log_std；动作经 tanh 限制在 [-1,1]（归一化动作空间）。
  Critic : 输入同一隐特征 (B, F)，输出状态价值标量 V (B,)。

接口对齐约定：
  - feat_dim 必须严格等于世界模型的 det_dim + stoch_dim（基线 = 200 + 48 = 248）。
    build_from_world_model(wm) 从 WorldModel 实例自动读取，避免维度错配。
  - 网络输出动作 ∈ [-1,1]，与数据集归一化动作同尺度；
    ActionScaler.to_env() 负责推理时映射回环境真实动作范围
    （Panda PushCube: 位置 [-0.02,0.02]，夹爪 [0,1]）。

重参数化采样在 sample() 内完成：a = tanh(mu_raw + sigma * eps)，eps ~ N(0,I)。
tanh 既保证动作落在 [-1,1]，又保留可微梯度，是梯度沿
"奖励 → 动作 → 策略参数"完整回传（想象训练可微的关键）的前提。
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class GaussianActor(nn.Module):
    """隐特征 -> 4 维归一化动作的高斯策略。

    forward(feat) 返回 (mu, log_std)，形状均为 (B, act_dim)：
      mu      : tanh 压缩后的动作分布均值，∈ [-1,1]；
      log_std : clamp 到 [log_std_min, log_std_max]，保证数值稳定。

    sample(feat) 在 tanh 之前的潜变量空间做重参数化再压缩到 [-1,1]，
    保证采样动作与 forward 的均值语义一致且梯度光滑。
    """

    def __init__(self, feat_dim: int, act_dim: int = 4,
                 hidden: Sequence[int] = (256, 256),
                 log_std_min: float = LOG_STD_MIN,
                 log_std_max: float = LOG_STD_MAX):
        super().__init__()
        self.feat_dim, self.act_dim = int(feat_dim), int(act_dim)
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        layers, prev = [], self.feat_dim
        for h in hidden:
            layers += [nn.Linear(prev, int(h)), nn.SiLU()]
            prev = int(h)
        self.trunk = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(prev, self.act_dim)
        self.log_std_layer = nn.Linear(prev, self.act_dim)
        # 输出层小初始化：避免初始标准差过大导致早期采样接近纯随机游走
        for layer in (self.mu_layer, self.log_std_layer):
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.zeros_(layer.bias)

    def _raw(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"Actor 输入特征维度应为 {self.feat_dim}，得到 {feat.shape[-1]}")
        h = self.trunk(feat)
        raw_mu = self.mu_layer(h)
        log_std = self.log_std_layer(h).clamp(self.log_std_min, self.log_std_max)
        return raw_mu, log_std

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_mu, log_std = self._raw(feat)
        return torch.tanh(raw_mu), log_std

    def sample(self, feat: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """重参数化采样。deterministic=True 时直接取均值 tanh(mu_raw)。"""
        raw_mu, log_std = self._raw(feat)
        if deterministic:
            return torch.tanh(raw_mu)
        std = log_std.exp()
        eps = torch.randn_like(raw_mu)
        return torch.tanh(raw_mu + std * eps)

    def log_prob(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """动作的近似高斯对数概率（逐维求和）。

        在 tanh 前的潜变量空间计算，未含 tanh 雅可比修正，供策略梯度类算法近似使用。
        严格实现可在后续训练阶段补充 atanh(action) 与 log(1 - tanh^2) 修正项。
        """
        raw_mu, log_std = self._raw(feat)
        std = log_std.exp()
        eps = torch.randn_like(raw_mu)
        raw_action = raw_mu + std * eps
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, device=feat.device))
        z = (raw_action - raw_mu) / (std + 1e-6)
        return (-0.5 * (z * z + 2.0 * log_std + log_two_pi)).sum(dim=-1)


class ValueCritic(nn.Module):
    """隐特征 -> 状态价值 V (B,)。"""

    def __init__(self, feat_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.feat_dim = int(feat_dim)
        layers, prev = [], self.feat_dim
        for h in hidden:
            layers += [nn.Linear(prev, int(h)), nn.SiLU()]
            prev = int(h)
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"Critic 输入特征维度应为 {self.feat_dim}，得到 {feat.shape[-1]}")
        return self.net(feat).squeeze(-1)   # (B,)


@dataclass
class ActionScaler:
    """归一化动作 [-1,1] 与环境真实动作范围的双向映射。

    公式（逐维）：
      norm = (a - low) / span * 2 - 1
      a    = (norm + 1) / 2 * span + low
    其中 span = high - low，high == low 的维度用 1 代替以避免除零。
    """

    low: torch.Tensor
    high: torch.Tensor

    @classmethod
    def from_arrays(cls, low, high, device=None) -> "ActionScaler":
        l = torch.as_tensor(low, dtype=torch.float32, device=device)
        h = torch.as_tensor(high, dtype=torch.float32, device=device)
        if l.shape != h.shape:
            raise ValueError(f"low/high 维度不一致：{l.shape} vs {h.shape}")
        return cls(low=l, high=h)

    @classmethod
    def from_hdf5(cls, hdf5_path: str, device=None) -> "ActionScaler":
        import h5py                      # 延迟导入，使模型模块本身不依赖 h5py
        import numpy as np
        with h5py.File(hdf5_path, "r") as f:
            low = np.asarray(f.attrs["action_low"], dtype=np.float32)
            high = np.asarray(f.attrs["action_high"], dtype=np.float32)
        return cls.from_arrays(low, high, device=device)

    def _span(self) -> torch.Tensor:
        return torch.where(self.high == self.low,
                           torch.ones_like(self.high), self.high - self.low)

    def to_env(self, norm_action: torch.Tensor) -> torch.Tensor:
        """[-1,1] 归一化动作 -> 环境真实动作（推理落地用）。"""
        span = self._span().to(norm_action.device)
        low = self.low.to(norm_action.device)
        return (norm_action + 1.0) / 2.0 * span + low

    def from_env(self, action: torch.Tensor) -> torch.Tensor:
        """环境真实动作 -> [-1,1] 归一化动作（数据集口径）。"""
        span = self._span().to(action.device)
        low = self.low.to(action.device)
        return ((action - low) / span * 2.0 - 1.0).clamp(-1.0, 1.0)


class ActorCritic(nn.Module):
    """组合 Actor + Critic，共享同一隐特征输入，便于统一保存/加载。"""

    def __init__(self, feat_dim: int, act_dim: int = 4,
                 hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.feat_dim, self.act_dim = int(feat_dim), int(act_dim)
        self.actor = GaussianActor(self.feat_dim, self.act_dim, hidden)
        self.critic = ValueCritic(self.feat_dim, hidden)

    def act(self, feat: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return self.actor.sample(feat, deterministic=deterministic)

    def values(self, feat: torch.Tensor) -> torch.Tensor:
        return self.critic(feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_actor_critic(feat_dim: int, act_dim: int = 4,
                       hidden: Sequence[int] = (256, 256)) -> ActorCritic:
    """按显式维度构建策略网络。"""
    return ActorCritic(feat_dim, act_dim, hidden)


def build_from_world_model(wm, act_dim: int = 4,
                           hidden: Sequence[int] = (256, 256)) -> ActorCritic:
    """从 WorldModel 实例读取 feat_dim 构建，严格对齐 RSSM 隐状态维度。"""
    return ActorCritic(wm.feat_dim, act_dim, hidden)


if __name__ == "__main__":
    torch.manual_seed(0)
    ac = ActorCritic(248)
    feat = torch.randn(6, 248)
    mu, log_std = ac.actor(feat)
    a = ac.actor.sample(feat)
    v = ac.values(feat)
    assert mu.shape == (6, 4) and a.shape == (6, 4)
    assert mu.abs().max() <= 1.0 + 1e-6 and a.abs().max() <= 1.0 + 1e-6
    assert v.shape == (6,)
    scaler = ActionScaler.from_arrays([-0.02] * 3 + [0.0], [0.02] * 3 + [1.0])
    real = scaler.to_env(a)
    assert real[:, :3].abs().max() <= 0.02 + 1e-6
    assert real[:, 3].min() >= -1e-6 and real[:, 3].max() <= 1.0 + 1e-6
    print(f"ACTOR_CRITIC_OK  mu={tuple(mu.shape)} act={tuple(a.shape)} v={tuple(v.shape)} "
          f"params={ac.count_parameters():,} real_pos_max={real[:, :3].abs().max():.4f} "
          f"real_grip=[{real[:, 3].min():.3f},{real[:, 3].max():.3f}]")
