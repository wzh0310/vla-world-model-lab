# -*- coding: utf-8 -*-
"""奖励预测器模块（模块解耦后独立）：RewardHead + ValueHead。

- RewardHead：输入 RSSM/MLP 拼接隐状态特征 (B, feat_dim)，回归单步奖励（归一化尺度）。
- ValueHead： 输入同样的隐状态特征，预测状态价值 V(s_t)，用于折扣回报回归与后续想象策略。

两个头均继承 nn.Sequential，权重 key 形如 reward_head.0.weight / value_head.2.bias，
与原训练脚本内联的 nn.Sequential 完全一致——历史 checkpoint 可无缝加载。

该模块从原 models/heads.py 拆出；原 heads.py 保留 RewardHead/ValueHead 的转发导入。
"""
from typing import Tuple

import torch
import torch.nn as nn


class _BaseHead(nn.Sequential):
    """MLP 回归头基类：Linear->SiLU->Linear->SiLU->Linear(1)。

    继承 nn.Sequential 保证 state_dict key 与旧版内联实现一致（0/2/4 为 Linear）。
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


class ValueHead(_BaseHead):
    """状态价值头：隐状态特征 -> 状态价值标量 (B, 1)。"""


if __name__ == "__main__":
    torch.manual_seed(0)
    rh, vh = RewardHead(264), ValueHead(264)
    feat = torch.randn(5, 264)
    assert rh(feat).shape == (5, 1) and vh(feat).shape == (5, 1)
    assert set(rh.state_dict().keys()) == {"0.weight", "0.bias", "2.weight", "2.bias",
                                           "4.weight", "4.bias"}
    print("REWARD_PREDICTOR_OK")
