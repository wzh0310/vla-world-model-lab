# -*- coding: utf-8 -*-
"""模型工厂：按配置统一构建动力学与预测头，集中消除脚本内的硬编码分支。

支持两种动力学（消融对照）：
  - "rssm": 循环确定性态 + 随机隐状态（DreamerV2 风格，默认）
  - "mlp" : 纯前馈 MLP 动力学（无循环、无随机），作为长程预测消融基线

对外函数：
  build_dynamics(dynamics, feat_dim, action_dim, embed_dim, stoch_dim, det_dim, hidden_dim)
      -> (nn.Module, feat_dim)
  build_reward_heads(feat_dim, hidden=(256,128)) -> (RewardHead, ValueHead)
"""
import torch.nn as nn

from models.rssm import RSSM
from models.mlp_dynamics import MLPDynamics
from models.reward_predictor import RewardHead, ValueHead

SUPPORTED = ("rssm", "mlp")


def build_dynamics(dynamics: str, feat_dim: int, action_dim: int, embed_dim: int,
                   stoch_dim: int, det_dim: int, hidden_dim: int = 256,
                   min_std: float = 0.1):
    """构建动力学模块，返回 (module, feat_dim)。

    rssm 的 feat_dim = det_dim + stoch_dim；mlp 的 feat_dim 直接取传入值
    （调用方保证两者一致，通常为 det_dim + stoch_dim）。
    """
    kind = str(dynamics).lower()
    if kind not in SUPPORTED:
        raise ValueError(f"未知动力学类型 {dynamics}，可选 {SUPPORTED}")
    if kind == "rssm":
        module = RSSM(stoch_dim=stoch_dim, det_dim=det_dim, action_dim=action_dim,
                      hidden_dim=hidden_dim, embed_dim=embed_dim, min_std=min_std)
        return module, det_dim + stoch_dim
    module = MLPDynamics(feat_dim=feat_dim, action_dim=action_dim,
                         embed_dim=embed_dim, min_std=min_std)
    return module, feat_dim


def build_reward_heads(feat_dim: int, hidden=(256, 128)):
    return RewardHead(feat_dim, hidden=hidden), ValueHead(feat_dim, hidden=hidden)
