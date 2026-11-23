# -*- coding: utf-8 -*-
"""RSSM（Recurrent State Space Model）动力学模型骨架 —— DreamerV2 风格。

核心数据流（每一步 t）：
    observe(prev_state, action, embed_t)   : 用真实图像嵌入做后验推断
      h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])           # 确定性隐藏态
      post_params = MLP(h_t, embed_t) -> (mu, logvar)  # 后验 q(z_t | h_t, x_t)
      z_t = tanh(mu + eps*sigma)                        # 随机隐状态 (tanh 限制范围)
    imagine_step(state, action)            : 无真实图像，纯动力学先验预测一步
      h' = GRU(h, [z, a])                               # 复用同一个 GRU
      prior_params = MLP(h') -> (mu, logvar)            # 先验 p(z' | h')
      z' = tanh(mu + eps*sigma)
    imagine_rollout(state, actions)        : 开环多步想象（本阶段新增）
      从初始状态出发，仅用动作序列自回归展开 N 步，全程不观测真实图像。
      与 observe 的 teacher-forcing 不同，误差会随步长累积，是"世界模型内
      想象训练"的核心接口（后续 actor-critic 在想象轨迹上学习）。
    decode_rollout(states, decoder)        : 批量解码想象状态序列为图像序列
      把 N 步特征一次性送入 decoder，返回 (N, B, C, H, W) 预测视频帧。
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

    def imagine_step(self, state: RSSMState, action: torch.Tensor,
                     sample: bool = True) -> Tuple[RSSMState, Dict[str, torch.Tensor]]:
        """纯动力学先验预测一步（无真实图像，想象 rollout 用）。

        sample=True  从先验高斯重参数化采样（默认，保留随机性）；
        sample=False 直接用先验均值 tanh(mu)，得到确定性预测，便于复现与调试。
        """
        det = self._transition(state, action)
        prior_mu, prior_logvar = self.prior(det).chunk(2, dim=-1)
        if sample:
            stoch = self._sample(prior_mu, prior_logvar, self.min_std)
        else:
            stoch = torch.tanh(prior_mu)
        return RSSMState(det=det, stoch=stoch), {"prior_mu": prior_mu, "prior_logvar": prior_logvar}

    def imagine_rollout(self, initial_state: RSSMState, actions: torch.Tensor,
                        sample: bool = True) -> Tuple[list, list]:
        """开环多步想象：从初始状态 + 动作序列生成 N 步隐状态，全程不接入真实图像。

        这是"世界模型内想象训练"的核心接口：rollout 过程中不调用 observe、
        不读取任何真实帧嵌入，只反复调用 imagine_step 自回归推进。因此它能生成
        任意步长的预测轨迹，但误差会随步长累积（区别于 teacher-forcing 单步预测）。

        参数：
          initial_state: RSSMState，batch=(B,*)；通常由真实首帧编码后的后验状态充当起点。
          actions: 动作序列，支持两种形状：
                   (B, N, A) 批次形式，或 (N, A) 单样本形式（B>1 时自动广播到各 batch）。
                   也接受 (A,) 单步动作。
          sample: 是否从先验分布采样；False 用先验均值（确定性，便于复现）。

        返回：
          states: List[RSSMState]，长度 N；第 k 项是预测的第 k+1 步状态。
          infos:  List[dict]，长度 N；各项含该步先验 mu / logvar。
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32,
                                      device=initial_state.det.device)
        if actions.ndim == 1:
            actions = actions.reshape(1, 1, -1)     # (A,) -> (1,1,A)
        elif actions.ndim == 2:
            actions = actions.unsqueeze(0)          # (N,A) -> (1,N,A)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"动作维度应为 {self.action_dim}，得到 {actions.shape[-1]}")
        B = initial_state.det.shape[0]
        if actions.shape[0] == 1 and B > 1:
            actions = actions.expand(B, -1, -1)
        if actions.shape[0] != B:
            raise ValueError(f"动作 batch={actions.shape[0]} 与状态 batch={B} 不一致")
        N = actions.shape[1]
        states, infos = [], []
        s = initial_state
        for k in range(N):
            s, info = self.imagine_step(s, actions[:, k], sample=sample)
            states.append(s)
            infos.append(info)
        return states, infos

    def get_feature(self, state: RSSMState) -> torch.Tensor:
        """确定性态 + 随机态拼接 -> decoder / reward head 的输入特征。"""
        return state.features()

    def decode_rollout(self, states: list, decoder: nn.Module) -> torch.Tensor:
        """批量解码想象状态序列为图像序列（imagine_rollout 的配套解码函数）。

        将 N 步状态的拼接特征一次性送入 decoder，避免逐步解码的 Python 循环开销。
        decoder 需输出 (N*B, C, H, W) 形式的图像张量（与项目 ConvDecoder 一致）。
        返回张量形状 (N, B, C, H, W)，第 k 项为第 k+1 步的预测图像。
        """
        if len(states) == 0:
            raise ValueError("states 不能为空")
        feats = torch.stack([self.get_feature(s) for s in states], dim=0)  # (N,B,F)
        N, B, F = feats.shape
        out = decoder(feats.reshape(N * B, F))
        if out.ndim < 3:
            raise ValueError("decoder 需输出 (batch, C, H, W) 图像张量，"
                             f"实际得到 {tuple(out.shape)}")
        return out.reshape(N, B, *out.shape[-3:])

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

    # 开环多步 rollout：只给动作序列，全程不接真实图像
    acts_seq = torch.randn(B, 10, ACT)
    states, infos = rssm.imagine_rollout(s1, acts_seq)
    assert len(states) == 10 and len(infos) == 10
    assert all(x.det.shape == (B, 200) and x.stoch.shape == (B, 64) for x in states)
    assert all(x.stoch.min() >= -1 and x.stoch.max() <= 1 for x in states)
    fake_dec = nn.Sequential(nn.Linear(feat.shape[-1], 3 * 32 * 32),
                             nn.Unflatten(1, (3, 32, 32)))
    imgs = rssm.decode_rollout(states, fake_dec)
    assert imgs.shape == (10, B, 3, 32, 32), imgs.shape
    # (N,A) 单样本形式在 B>1 时应广播
    states2, _ = rssm.imagine_rollout(s1, torch.randn(10, ACT), sample=False)
    assert len(states2) == 10 and states2[0].det.shape == (B, 200)
    print(f"开环 rollout: {len(states)} 步, 批量解码图像 {tuple(imgs.shape)}")
    print(f"RSSM_OK  参数量={rssm.count_parameters():,}")
