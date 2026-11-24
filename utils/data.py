# -*- coding: utf-8 -*-
"""数据加载工具统一入口。

实现位于 ``datasets/``：
  - ``EpisodicDataset``   —— 20 步序列 / 64×64 / 动作与奖励归一化
  - ``get_dataloader``    —— 常规 DataLoader 封装
  - ``GPUSequenceSampler``—— GPU 显存预加载采样（约 80× 加速）

本模块只做路径引导与再导出，避免同一份数据集逻辑出现多个副本。
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "datasets")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from episodic_dataset import EpisodicDataset, get_dataloader  # noqa: E402,F401
from fast_loader import GPUSequenceSampler  # noqa: E402,F401
