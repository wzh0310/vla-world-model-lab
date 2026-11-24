# -*- coding: utf-8 -*-
"""纯 MLP 动力学基线（消融对照）：无 GRU 循环、无随机隐状态。

设计目的：与 RSSM 做受控对比，验证"循环确定性态 + 随机隐状态"是否真的带来
长程预测优势。MLP 基线保留相同的对外接口与特征维度，唯一区别是动力学本体：

  - RSSM:      h_t = GRU(h_{t-1}, [z_{t-1}, a_t]); z_t ~ q(z|h,embed)   （循环 + 随机）
  - MLP 基线:  feat_t = MLP([feat_{t-1}, a_t, embed_t])                 （纯前馈、确定性）

记忆仅由上一时刻特征向量本身携带（无独立循环态），开环想象时无真实 embed，
退化为 feat_{t+1} = MLP_img([feat_t, a_t])，误差会随步长累积。

接口与 RSSM 完全一致：initial_state / observe / imagine_step / imagine_rollout /
get_feature / decode_rollout / min_std / count_parameters，故可作为 drop-in 替换，
训练脚本、评估脚本、WorldModel 无需针对动力学分支写特例。

为兼容训练循环的 KL 计算，observe/imagine_step 返回的 info 令 prior==post，
使 KL 恒为 0（调用方对 MLP 基线传 --kl-scale 0 即可完全关闭 KL 项）。
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class MLPState:
    """MLP 动力学单步状态：仅一个确定性特征向量（与 RSSM 特征同维）。"""
    feat: torch.Tensor  # (B, feat_dim)

    def features(self) -> torch.Tensor:
        return self.feat


class MLPDynamics(nn.Module):
    """纯 MLP 动力学：状态 = feat_dim 维特征，动作驱动前馈转移。"""

    def __init__(self, feat_dim: int = 264, action_dim: int = 4, embed_dim: int = 64,
                 hidden_dim: int = 512, min_std: float = 0.1):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)
        self.embed_dim = int(embed_dim)
        self.min_std = float(min_std)  # 仅为兼容训练脚本 KL 接口
        self.stoch_dim = 0             # 便于评估脚本读取配置
        self.det_dim = int(feat_dim)

        self.obs_mlp = nn.Sequential(
            nn.Linear(self.feat_dim + action_dim + embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.feat_dim),
        )
        self.img_mlp = nn.Sequential(
            nn.Linear(self.feat_dim + action_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.feat_dim),
        )

    def initial_state(self, batch_size: int, device=None) -> MLPState:
        return MLPState(feat=torch.zeros(batch_size, self.feat_dim, device=device))

    @staticmethod
    def _info(feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """prior==post => KL=0，兼容训练脚本的 kl_gaussian 接口。"""
        z = torch.zeros_like(feat)
        return {"prior_mu": feat, "prior_logvar": z,
                "post_mu": feat, "post_logvar": z}

    def observe(self, state: MLPState, action: torch.Tensor,
                embed: torch.Tensor) -> Tuple[MLPState, Dict[str, torch.Tensor]]:
        if embed.ndim == 1:
            embed = embed.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        feat = self.obs_mlp(torch.cat([state.feat, action, embed], dim=-1))
        return MLPState(feat=feat), self._info(feat)

    def imagine_step(self, state: MLPState, action: torch.Tensor,
                     sample: bool = True) -> Tuple[MLPState, Dict[str, torch.Tensor]]:
        if action.ndim == 1:
            action = action.unsqueeze(0)
        feat = self.img_mlp(torch.cat([state.feat, action], dim=-1))
        return MLPState(feat=feat), self._info(feat)

    def imagine_rollout(self, initial_state: MLPState, actions: torch.Tensor,
                        sample: bool = True) -> Tuple[List[MLPState], List[dict]]:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32,
                                      device=initial_state.feat.device)
        if actions.ndim == 1:
            actions = actions.reshape(1, 1, -1)
        elif actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"动作维度应为 {self.action_dim}，得到 {actions.shape[-1]}")
        B = initial_state.feat.shape[0]
        if actions.shape[0] == 1 and B > 1:
            actions = actions.expand(B, -1, -1)
        if actions.shape[0] != B:
            raise ValueError(f"动作 batch={actions.shape[0]} 与状态 batch={B} 不一致")
        states, infos = [], []
        s = initial_state
        for k in range(actions.shape[1]):
            s, info = self.imagine_step(s, actions[:, k], sample=sample)
            states.append(s)
            infos.append(info)
        return states, infos

    def get_feature(self, state: MLPState) -> torch.Tensor:
        return state.features()

    def decode_rollout(self, states: List[MLPState], decoder: nn.Module) -> torch.Tensor:
        if len(states) == 0:
            raise ValueError("states 不能为空")
        feats = torch.stack([self.get_feature(s) for s in states], dim=0)  # (N,B,F)
        N, B, F = feats.shape
        out = decoder(feats.reshape(N * B, F))
        return out.reshape(N, B, *out.shape[-3:])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    F, A, E = 264, 4, 64
    dyn = MLPDynamics(feat_dim=F, embed_dim=E)
    s = dyn.initial_state(3)
    a = torch.randn(3, A)
    e = torch.randn(3, E)
    s1, info = dyn.observe(s, a, e)
    assert dyn.get_feature(s1).shape == (3, F)
    acts_seq = torch.randn(3, 10, A)
    states, _ = dyn.imagine_rollout(s1, acts_seq, sample=False)
    assert len(states) == 10 and states[0].feat.shape == (3, F)
    fake_dec = nn.Sequential(nn.Linear(F, 3 * 32 * 32), nn.Unflatten(1, (3, 32, 32)))
    imgs = dyn.decode_rollout(states, fake_dec)
    assert imgs.shape == (10, 3, 3, 32, 32)
    print("MLP_DYNAMICS_OK", tuple(imgs.shape))
