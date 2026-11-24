# -*- coding: utf-8 -*-
"""预测头兼容层（模块解耦后）。

RewardHead / ValueHead 已拆分到独立模块 models/reward_predictor.py。
本文件保留转发导入（向后兼容所有历史脚本与权重）：
    from models.heads import RewardHead, ValueHead
权重 key（reward_head.0.weight / value_head.4.bias 等）保持不变。
"""
from models.reward_predictor import RewardHead, ValueHead, _BaseHead

__all__ = ["RewardHead", "ValueHead", "_BaseHead"]
