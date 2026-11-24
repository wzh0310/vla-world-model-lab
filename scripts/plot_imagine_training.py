#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""绘制想象策略训练曲线（英文标签，容器无中文字体）。"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="多组 train_summary.json，如 v1 v2 对照")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.summaries]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for path, label in zip(args.summaries, args.labels or labels):
        d = json.load(open(path, encoding="utf-8"))
        h = d["history"]
        ep = [x["epoch"] for x in h]
        rew = [x["imagined_reward"] for x in h]
        vloss = [x["value_loss"] for x in h]
        probe = [(x["epoch"], x["probe_return"]) for x in h if "probe_return" in x]
        axes[0].plot(ep, rew, lw=1.3, alpha=0.5)
        axes[1].plot(ep, vloss, lw=1.3, alpha=0.5)
        if probe:
            axes[2].plot([p[0] for p in probe], [p[1] for p in probe], lw=1.8, label=label, marker="o", ms=2)
    axes[0].set_title("Imagined reward per batch"); axes[0].set_xlabel("Epoch")
    axes[1].set_title("Value (Huber) loss"); axes[1].set_xlabel("Epoch")
    axes[2].set_title("Probe return (fixed starts)"); axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print("PLOT_OK", args.out)


if __name__ == "__main__":
    main()
