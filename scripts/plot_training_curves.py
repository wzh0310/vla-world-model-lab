#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练曲线可视化：从 TensorBoard events 读三条核心损失 + 周期性评估曲线。

运行容器环境：vla。产出 outputs/wm_mixtrain/training_curves.png。
图例用英文（容器无中文字体）。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # noqa: F401  确保 event parser 可用
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "/data/wzh/robot/vla-world-model-lab"
LOGDIR = os.path.join(ROOT, "outputs/wm_mixtrain/logs")
OUT = os.path.join(ROOT, "outputs/wm_mixtrain/training_curves.png")


def read_scalar(tag):
    acc = EventAccumulator(LOGDIR)
    acc.Reload()
    try:
        evs = acc.Scalars(tag)
    except KeyError:
        return np.array([]), np.array([])
    return np.array([e.step for e in evs]), np.array([e.value for e in evs])


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 三条核心损失曲线
    for tag, label, color in [
        ("loss/recon", "Reconstruction loss", "#d62728"),
        ("loss/kl", "KL divergence", "#1f77b4"),
        ("loss/reward", "Reward prediction error", "#2ca02c"),
    ]:
        s, v = read_scalar(tag)
        if len(s):
            axes[0, 0].plot(s, v, label=label, color=color, lw=1.5, alpha=0.85)
    axes[0, 0].set_xlabel("Train step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Core training losses (3 curves)")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # 总损失 + 价值损失
    for tag, label, color in [
        ("loss/total", "Total loss", "#9467bd"),
        ("loss/value", "Value loss", "#ff7f0e"),
    ]:
        s, v = read_scalar(tag)
        if len(s):
            axes[0, 1].plot(s, v, label=label, color=color, lw=1.5, alpha=0.85)
    axes[0, 1].set_xlabel("Train step")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Total & value loss")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # 学习率分组
    for tag, label, color in [
        ("train/lr", "LR (unified/enc)", "#1f77b4"),
    ]:
        s, v = read_scalar(tag)
        if len(s):
            axes[1, 0].plot(s, v, label=label, color=color, lw=2)
    axes[1, 0].set_xlabel("Train step")
    axes[1, 0].set_ylabel("Learning rate")
    axes[1, 0].set_title("Cosine LR schedule (enc 1e-4 / dyn 3e-4 grouped)")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)

    # 周期性评估曲线
    for tag, label, color in [
        ("eval/ssim20", "SSIM@20 (target>=0.6)", "#2ca02c"),
    ]:
        s, v = read_scalar(tag)
        if len(s):
            axes[1, 1].plot(s, v, marker="o", label=label, color=color, lw=2)
            axes[1, 1].axhline(0.6, ls="--", color="gray", alpha=0.6, label="target 0.6")
    axes[1, 1].set_xlabel("Train step")
    axes[1, 1].set_ylabel("SSIM")
    axes[1, 1].set_title("Periodic open-loop evaluation (20 steps)")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("World Model Full Training (mixed dataset) (5000 steps, mixed dataset)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"[PLOT] saved {OUT}")


if __name__ == "__main__":
    main()
