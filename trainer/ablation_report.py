#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""消融实验汇总：读取 4 组多步评估 JSON，生成对比表格(CSV/Markdown)与折线图。

4 组：
  base    有 KL, stoch=64（基线，复用 wm_tune1）
  stoch16 有 KL, stoch=16
  stoch32 有 KL, stoch=32
  nokl    无 KL, stoch=64

输出（写入 --out-dir，默认 outputs/ablation_report/）：
  ablation_metrics.csv          关键步 1/5/10/20 的 PSNR/SSIM/奖励MAE 汇总表
  ablation_metrics.md           同上 Markdown 表格（直接贴报告）
  psnr_by_dim.png               维度消融 PSNR 随步长曲线（16/32/64）
  psnr_by_kl.png                KL 消融 PSNR 随步长曲线（有/无）
  ssim_compare.png              两组消融 SSIM 随步长曲线
  stability_bar.png             长序列稳定性柱状图（1->20 PSNR 衰减 / 末段 SSIM）

用法：python trainer/ablation_report.py
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 图表标签用英文（容器无中文字体）；表格/文档保留中文说明
LABELS = {
    "base": "Base (KL, stoch64)",
    "stoch16": "stoch16 (KL)",
    "stoch32": "stoch32 (KL)",
    "nokl": "No-KL (stoch64)",
}
DIRS = {
    "base": "outputs/abl_base",
    "stoch16": "outputs/abl_stoch16",
    "stoch32": "outputs/abl_stoch32",
    "nokl": "outputs/abl_nokl",
}


def parse_args():
    ap = argparse.ArgumentParser(description="消融实验汇总与可视化")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/ablation_report"))
    return ap.parse_args()


def load_all():
    data = {}
    for key, d in DIRS.items():
        p = os.path.join(ROOT, d, "metrics", "multistep_full_eval.json")
        if not os.path.exists(p):
            print(f"[WARN] 缺少 {p}，跳过", flush=True)
            continue
        with open(p, encoding="utf-8") as f:
            data[key] = json.load(f)
    return data


def key_step(s, k, field):
    return s.get("key_steps", {}).get(f"step{k}", {}).get(field)


def write_tables(data, out_dir):
    """关键步指标汇总表：每行 = 模型，列 = 各步 PSNR/SSIM/奖励MAE。"""
    steps = [1, 5, 10, 20]
    header = ["模型", "stoch", "KL"]
    for k in steps:
        header += [f"PSNR@{k}(dB)", f"SSIM@{k}"]
    header += ["奖励MAE@20(norm)", "PSNR衰减(1->20)"]
    rows = []
    for key in ["base", "stoch16", "stoch32", "nokl"]:
        if key not in data:
            continue
        s = data[key]
        stoch = s.get("stoch")
        kl = "无" if key == "nokl" else "有"
        row = [LABELS[key], stoch, kl]
        for k in steps:
            row += [key_step(s, k, "psnr_db"), key_step(s, k, "ssim")]
        row += [key_step(s, 20, "reward_mae_norm"), s.get("psnr_drop_1to20")]
        rows.append(row)

    # CSV
    csv_path = os.path.join(out_dir, "ablation_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else round(v, 4) if isinstance(v, float) else v for v in r])

    # Markdown
    md_path = os.path.join(out_dir, "ablation_metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(str(h) for h in header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            cells = []
            for v in r:
                if v is None:
                    cells.append("-")
                elif isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"[REPORT] 表格: {csv_path} / {md_path}", flush=True)
    return rows


def plot_psnr_curves(data, keys, colors, title, save_path, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key in keys:
        if key not in data:
            continue
        s = data[key]
        curve = s.get("psnr_mean_by_step", [])
        xs = np.arange(1, len(curve) + 1)
        ax.plot(xs, curve, marker="o", ms=3.5, lw=2, color=colors[key], label=LABELS[key])
    ax.set_xlabel("Open-loop prediction step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"[REPORT] 折线图: {save_path}", flush=True)


def plot_ssim(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # 维度消融
    for key in ["base", "stoch16", "stoch32"]:
        if key in data:
            c = data[key].get("ssim_mean_by_step", [])
            axes[0].plot(np.arange(1, len(c) + 1), c, marker="o", ms=3, lw=1.8,
                         color={"base": "#d62728", "stoch16": "#2ca02c", "stoch32": "#1f77b4"}[key],
                         label=LABELS[key])
    axes[0].set_title("SSIM - latent dimension ablation")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("SSIM")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1)
    # KL 消融
    for key in ["base", "nokl"]:
        if key in data:
            c = data[key].get("ssim_mean_by_step", [])
            axes[1].plot(np.arange(1, len(c) + 1), c, marker="o", ms=3, lw=1.8,
                         color={"base": "#d62728", "nokl": "#9467bd"}[key], label=LABELS[key])
    axes[1].set_title("SSIM - KL ablation")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("SSIM")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8); axes[1].set_ylim(0, 1)
    fig.tight_layout()
    p = os.path.join(out_dir, "ssim_compare.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[REPORT] 折线图: {p}", flush=True)


def plot_stability_bar(data, out_dir):
    keys = [k for k in ["base", "stoch16", "stoch32", "nokl"] if k in data]
    drop = [data[k].get("psnr_drop_1to20", 0) for k in keys]          # 越小越稳
    last_ssim = [key_step(data[k], 20, "ssim") or 0 for k in keys]   # 越大越稳
    x = np.arange(len(keys))
    w = 0.38
    fig, ax1 = plt.subplots(figsize=(9, 5))
    b1 = ax1.bar(x - w / 2, drop, w, color="#d62728", alpha=0.85, label="PSNR drop 1->20 (dB)")
    ax1.set_ylabel("PSNR drop (dB)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, last_ssim, w, color="#1f77b4", alpha=0.85, label="SSIM @ step20")
    ax2.set_ylabel("SSIM @ step20", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[k] for k in keys], fontsize=9)
    ax1.set_title("Long-horizon stability comparison")
    ax1.grid(axis="y", alpha=0.2)
    for b in list(b1) + list(b2):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "stability_bar.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[REPORT] 柱状图: {p}", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load_all()
    if not data:
        raise RuntimeError("没有任何评估结果 JSON，请先运行 ablation_eval.sh")
    write_tables(data, args.out_dir)
    plot_psnr_curves(data, ["base", "stoch16", "stoch32"],
                     {"base": "#d62728", "stoch16": "#2ca02c", "stoch32": "#1f77b4"},
                     "PSNR vs horizon - latent dimension ablation",
                     os.path.join(args.out_dir, "psnr_by_dim.png"))
    plot_psnr_curves(data, ["base", "nokl"],
                     {"base": "#d62728", "nokl": "#9467bd"},
                     "PSNR vs horizon - KL ablation",
                     os.path.join(args.out_dir, "psnr_by_kl.png"))
    plot_ssim(data, args.out_dir)
    plot_stability_bar(data, args.out_dir)
    # 汇总 JSON 便于文档引用
    with open(os.path.join(args.out_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: {"stoch": v.get("stoch"),
                       "psnr_step1": v.get("psnr_step1"),
                       "psnr_step20": v.get("psnr_step20"),
                       "psnr_drop_1to20": v.get("psnr_drop_1to20"),
                       "key_steps": v.get("key_steps")} for k, v in data.items()},
                  f, ensure_ascii=False, indent=2)
    print("[REPORT] OK", flush=True)


if __name__ == "__main__":
    main()
