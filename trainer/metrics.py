# -*- coding: utf-8 -*-
"""兼容转发层：指标实现已统一迁移至 ``utils/metrics.py``。

保留本文件是为了不破坏历史脚本中的 ``from trainer.metrics import ...`` 导入，
以及外部对旧路径的引用；新代码请直接导入 ``utils.metrics``。
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.metrics import *  # noqa: F401,F403
from utils.metrics import (  # noqa: F401
    PerceptualMetric,
    latent_drift_l2,
    mse_metric,
    psnr_from_mse,
    psnr_metric,
    reward_mae,
    ssim_metric,
    summarize_by_step,
)
