#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""消融结果可视化：读取统一评估 JSON，生成对比图表到 results/。

产出：
  dim_psnr.png       维度消融(16/32/48/64) PSNR 随步长曲线
  dim_ssim.png       维度消融 SSIM 随步长曲线
  dyn_psnr.png       RSSM vs MLP 动力学 PSNR 曲线
  dyn_ssim.png       RSSM vs MLP 动力学 SSIM 曲线
  stability_bar.png  各模型 PSNR 衰减 + 末段 SSIM 柱状对比
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

COLORS = {"stoch16": "#2ca02c", "stoch32": "#1f77b4", "stoch48": "#d62728",
          "stoch64": "#ff7f0e", "mlp": "#9467bd"}


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=os.path.join(ROOT, "results/abl_unified_metrics.json"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.metrics, encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    def curve(ax, name, field, color):
        s = data[name]
        c = s.get(field, [])
        # 维度组(名字本身已含 stoch)与动力学组采用不同标签，避免 "stoch16 (stoch16, rssm)" 冗余
        label = name if name.startswith("stoch") else f"{name} (stoch{s['stoch']}, {s['dynamics']})"
        ax.plot(np.arange(1, len(c) + 1), c, marker="o", ms=3.5, lw=2,
                color=color, label=label)
        return s

    dim_names = [n for n in ["stoch16", "stoch32", "stoch48", "stoch64"] if n in data]
    dyn_names = [n for n in ["stoch48", "mlp"] if n in data]

    # 维度 PSNR
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n in dim_names:
        curve(ax, n, "psnr_by_step", COLORS.get(n, "#333"))
    ax.set_xlabel("Open-loop step"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Latent dimension ablation - PSNR vs horizon")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "dim_psnr.png"), dpi=140); plt.close(fig)

    # 维度 SSIM
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n in dim_names:
        curve(ax, n, "ssim_by_step", COLORS.get(n, "#333"))
    ax.set_xlabel("Open-loop step"); ax.set_ylabel("SSIM")
    ax.set_title("Latent dimension ablation - SSIM vs horizon")
    ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "dim_ssim.png"), dpi=140); plt.close(fig)

    # 动力学 PSNR / SSIM
    for field, fn, ylabel in [("psnr_by_step", "dyn_psnr.png", "PSNR (dB)"),
                              ("ssim_by_step", "dyn_ssim.png", "SSIM")]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for n in dyn_names:
            curve(ax, n, field, COLORS.get(n, "#333"))
        ax.set_xlabel("Open-loop step"); ax.set_ylabel(ylabel)
        ax.set_title("Dynamics ablation - RSSM vs MLP")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        if field == "ssim_by_step":
            ax.set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, fn), dpi=140); plt.close(fig)

    # 稳定性柱状
    names = dim_names + ([n for n in dyn_names if n not in dim_names])
    drop = [data[n].get("psnr_drop_1to20", 0) for n in names]
    last_ssim = [data[n]["key_steps"].get("step20", {}).get("ssim", 0) for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    b1 = ax1.bar(x - w / 2, drop, w, color="#d62728", alpha=0.85, label="PSNR drop 1->20 (dB)")
    ax1.set_ylabel("PSNR drop (dB)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, last_ssim, w, color="#1f77b4", alpha=0.85, label="SSIM @ step20")
    ax2.set_ylabel("SSIM @ step20", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4"); ax2.set_ylim(0, 1)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_title("Long-horizon stability across ablations")
    for b in list(b1) + list(b2):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.2f}",
                 ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "stability_bar.png"), dpi=140); plt.close(fig)
    print(f"[PLOT] 图表已写入 {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
