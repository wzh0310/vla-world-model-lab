#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规范入口：多步开环预测可视化。

复用 ``scripts/compare_video.py`` 的完整实现：以首帧后验为起点、仅输入动作序列
展开 20 步想象，输出「真实帧 | 预测帧」并排动图、5/10/20 步关键帧三联图与
PSNR/SSIM 衰减曲线，统一写入 ``results/visualization/``。

用法（vla 环境）：
    python scripts/visualize_predictions.py \
        --ckpt outputs/wm_mixtrain/checkpoints/wm_best.pt \
        --data outputs/data/wm_dataset/test.h5 \
        --out-dir results/visualization

关键场景静态截图另见 ``demos/screenshots/``；单步预测 GIF 可由
``trainer/make_prediction_gif.py`` 生成。
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.compare_video import main  # noqa: E402


if __name__ == "__main__":
    main()
