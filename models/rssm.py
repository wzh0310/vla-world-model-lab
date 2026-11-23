# -*- coding: utf-8 -*-
"""RSSM（Recurrent State Space Model）动力学模型骨架 —— DreamerV2 风格。

核心数据流（每一步 t）：
    observe(prev_state, action, embed_t)   : 用真实图像嵌入做后验推断
      h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])           # 确定性隐藏态
      post_params = MLP(h_t, embed_t) -> (mu, logvar)  # 后验 q(z_t | h_t, x_t)
      z_t = tanh(mu + eps*sigma)                        # 随机隐状态 (tanh 限制范围)
    imagine_step(state, action)            : 无真实图像，纯动力学先验预测
      h' = GRU(h, [z, a])                               # 复用同一个 GRU
      prior_params = MLP(h') -> (mu, logvar)            # 先验 p(z' | h')
      z' = tanh(mu + eps*sigma)
    get_feature()                          : 拼接确定性态 + 随机态 -> decoder / reward head

设计要点：
- 确定性态 h 保证时序连贯（记忆），随机态 z 表达环境不确定性（可重参数化）。
- observe 产出后验（用于训练时对齐）与先验，供 KL 损失使用；imagine_step 只产出先验（推理/想象 rollout）。
- 随机态维度默认 64，与 AE 全量训练后的隐空间维度一致（但本骨架独立，不依赖 vae.py，方便后续替换编码器）。
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    """RSSM 单步状态容器。

    det:    (B, det_dim)   确定性 GRU 隐藏态
    stoch:  (B, stoch_dim) 随机隐状态（tanh 激活，范围 [-1,1]）
    """
    det: torch.Tensor
    stoch: torch.Tensor

    def features(self) -> torch.Tensor:
        """拼接为完整特征，供解码器与奖励头使用。"""
        return torch.cat([self.det, self.stoch], dim=-1)


class RSSM(nn.Module):
    """DreamerV2 风格 RSSM：确定性 GRU + 对角高斯随机态。"""

    def __init__(self, stoch_dim: int = 64, det_dim: int = 200,
                 action_dim: int = 4, hidden_dim: int = 256,
                 embed_dim: Optional[int] = None, min_std: float = 0.1):
        super().__init__()
        if min(stoch_dim, det_dim, action_dim, hidden_dim) < 1:
            raise ValueError("所有维度必须为正整数")
        if not 0.0 < min_std < 1.0:
            raise ValueError("min_std 必须在 (0,1) 内")
        self.stoch_dim, self.det_dim = int(stoch_dim), int(det_dim)
        self.action_dim, self.min_std = int(action_dim), float(min_std)
        self.embed_dim = embed_dim  # None 表示延迟推断（首次 observe 时按输入维度建立投影）

        # 确定性转移：GRU 输入 = [上一随机态, 上一动作]
        self.gru = nn.GRUCell(self.stoch_dim + self.action_dim, self.det_dim)
        # 后验/先验网络：输出 2*stoch_dim（mu 与 logvar）
        self.prior = nn.Sequential(nn.Linear(self.det_dim, hidden_dim), nn.SiLU(),
                                   nn.Linear(hidden_dim, 2 * self.stoch_dim))
        self.posterior = nn.Sequential(nn.Linear(self.det_dim + (embed_dim or 0), hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, 2 * self.stoch_dim))
        if embed_dim is not None:
            self._post_in = nn.Identity()  # 维度已固定
        else:
            self._post_in = None  # 延迟建立

    def _ensure_posterior_input(self, embed_dim: int):
        """首次 observe 时按图像嵌入维度重建后验网络（延迟推断输入维度）。"""
        if self._post_in is None:
            hidden = self.prior[0].out_features
            self.posterior = nn.Sequential(
                nn.Linear(self.det_dim + embed_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, 2 * self.stoch_dim)).to(self.prior[0].weight.device)
            self._post_in = nn.Identity()  # 标记已初始化
            self.embed_dim = embed_dim

    def initial_state(self, batch_size: int, device=None) -> RSSMState:
        """零初始状态。"""
        det = torch.zeros(batch_size, self.det_dim, device=device)
        stoch = torch.tanh(torch.zeros(batch_size, self.stoch_dim, device=device))
        return RSSMState(det=det, stoch=stoch)

    @staticmethod
    def _sample(mu: torch.Tensor, logvar: torch.Tensor, min_std: float) -> torch.Tensor:
        std = F.softplus(logvar) + min_std
        return torch.tanh(mu + torch.randn_like(mu) * std)

    def _transition(self, state: RSSMState, action: torch.Tensor) -> torch.Tensor:
        """复用 GRU 计算下一确定性态（observe / imagine_step 共用）。"""
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"动作维度应为 {self.action_dim}，得到 {action.shape[-1]}")
        x = torch.cat([state.stoch, action], dim=-1)
        return self.gru(x, state.det)

    def observe(self, state: RSSMState, action: torch.Tensor,
                embed: torch.Tensor) -> Tuple[RSSMState, Dict[str, torch.Tensor]]:
        """基于真实图像嵌入做一步后验推断（训练用）。

        返回 (new_state, info)，info 含先验 mu/logvar 与后验 mu/logvar（供 KL 损失）。
        """
        if embed.ndim == 1:
            embed = embed.unsqueeze(0)
        if self.embed_dim is None or self._post_in is None:
            self._ensure_posterior_input(embed.shape[-1])
        det = self._transition(state, action)
        prior_mu, prior_logvar = self.prior(det).chunk(2, dim=-1)
        post_mu, post_logvar = self.posterior(torch.cat([det, embed], dim=-1)).chunk(2, dim=-1)
        stoch = self._sample(post_mu, post_logvar, self.min_std)
        new_state = RSSMState(det=det, stoch=stoch)
        info = {"prior_mu": prior_mu, "prior_logvar": prior_logvar,
                "post_mu": post_mu, "post_logvar": post_logvar}
        return new_state, info

    def imagine_step(self, state: RSSMState, action: torch.Tensor) -> Tuple[RSSMState, Dict[str, torch.Tensor]]:
        """纯动力学先验预测一步（无真实图像，想象 rollout 用）。"""
        det = self._transition(state, action)
        prior_mu, prior_logvar = self.prior(det).chunk(2, dim=-1)
        stoch = self._sample(prior_mu, prior_logvar, self.min_std)
        return RSSMState(det=det, stoch=stoch), {"prior_mu": prior_mu, "prior_logvar": prior_logvar}

    def get_feature(self, state: RSSMState) -> torch.Tensor:
        """确定性态 + 随机态拼接 -> decoder / reward head 的输入特征。"""
        return state.features()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, EMBED, ACT = 3, 64, 4
    rssm = RSSM(stoch_dim=64, det_dim=200, action_dim=ACT, embed_dim=EMBED)
    s = rssm.initial_state(B)
    act = torch.randn(B, ACT)
    embed = torch.randn(B, EMBED)

    s1, info = rssm.observe(s, act, embed)
    feat = rssm.get_feature(s1)
    s2, info2 = rssm.imagine_step(s1, act)

    print(f"初始状态: det={tuple(s.det.shape)} stoch={tuple(s.stoch.shape)}")
    print(f"observe 后: det={tuple(s1.det.shape)} stoch={tuple(s1.stoch.shape)}")
    print(f"imagine 后: det={tuple(s2.det.shape)} stoch={tuple(s2.stoch.shape)}")
    print(f"拼接特征: {tuple(feat.shape)} = det(200) + stoch(64)")
    for k, v in info.items():
        print(f"  info[{k}]: {tuple(v.shape)}")
    assert feat.shape == (B, 264)
    assert s1.stoch.min() >= -1 and s1.stoch.max() <= 1, "随机态应经 tanh 限制在 [-1,1]"
    # 多样性与可复现性检查
    s1b, _ = rssm.observe(s, act, embed)
    assert not torch.equal(s1.stoch, s1b.stoch), "随机采样应产生不同隐状态"
    print(f"RSSM_OK  参数量={rssm.count_parameters():,}")
