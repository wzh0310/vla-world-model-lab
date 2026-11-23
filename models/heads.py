# -*- coding: utf-8 -*-
"""世界模型的两个预测头：奖励头与价值头。

设计要点：
- RewardHead：输入 RSSM 拼接隐状态特征 (B, det+stoch)，回归单步奖励（归一化尺度）。
- ValueHead： 输入同样的隐状态特征，预测状态价值 V(s_t)，为后续在想象轨迹上
              训练 actor-critic 做准备（价值目标当前用真实轨迹的折扣回报）。
- 两个头均继承 nn.Sequential，权重 key 形如 reward_head.0.weight / value_head.2.bias，
  与原训练脚本内联的 reward_head（nn.Sequential）完全一致——因此历史检查点可无缝
  加载，且 eval_single_step.py 等既有脚本按前缀切片加载时不会受影响。
- 隐藏层尺寸默认 (256, 128)，与原奖励头一致；可用构造参数覆盖。
"""
from typing import Tuple

import torch
import torch.nn as nn


class _BaseHead(nn.Sequential):
    """MLP 回归头基类：Linear->SiLU->Linear->SiLU->Linear(1)。

    继承 nn.Sequential 是关键：保证 state_dict 的 key 与旧版内联 nn.Sequential
    完全一致（0/2/4 为 Linear 层），从而兼容历史权重与既有加载脚本。
    """

    def __init__(self, feat_dim: int, hidden: Tuple[int, int] = (256, 128)):
        h1, h2 = int(hidden[0]), int(hidden[1])
        if feat_dim < 1 or h1 < 1 or h2 < 1:
            raise ValueError("特征维度与隐藏层维度必须为正整数")
        super().__init__(
            nn.Linear(feat_dim, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, 1),
        )


class RewardHead(_BaseHead):
    """奖励预测头：隐状态特征 -> 单步奖励标量 (B, 1)。"""

    pass


class ValueHead(_BaseHead):
    """状态价值头：隐状态特征 -> 状态价值标量 (B, 1)。

    训练阶段以真实轨迹的折扣回报为监督信号；后续想象训练中可作为 critic，
    估计想象状态的价值并计算优势（advantage）。
    """

    pass


if __name__ == "__main__":
    torch.manual_seed(0)
    B, F = 5, 264
    rh, vh = RewardHead(F), ValueHead(F)
    feat = torch.randn(B, F)
    r, v = rh(feat), vh(feat)
    print(f"RewardHead 输出: {tuple(r.shape)}, 参数量 {sum(p.numel() for p in rh.parameters()):,}")
    print(f"ValueHead  输出: {tuple(v.shape)}, 参数量 {sum(p.numel() for p in vh.parameters()):,}")
    # 权重 key 兼容性断言：必须与旧内联 nn.Sequential 的 0/2/4 一致
    assert set(rh.state_dict().keys()) == {"0.weight", "0.bias", "2.weight", "2.bias",
                                           "4.weight", "4.bias"}
    assert set(vh.state_dict().keys()) == set(rh.state_dict().keys())
    print("HEADS_OK  权重 key 与旧版 reward_head 兼容")
