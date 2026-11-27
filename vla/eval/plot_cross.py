# -*- coding: utf-8 -*-
"""绘制跨任务与横向对比柱状图（英文标签避免字体问题）。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("results/vla_cross_task")
GEN = Path("results/vla_generalization")


def load(p):
    return json.load(open(p, encoding="utf-8"))


def barfig(values, labels, title, ylabel, out, colors, ylim=(0, 1.05)):
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=140)
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11); ax.set_title(title, fontsize=13)
    ax.set_ylim(*ylim); ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    base = load(BASE / "vla_baseline_raw.json")
    few = load(BASE / "vla_fewshot_raw.json")
    act = load(BASE / "act_raw.json")
    gc = load(GEN / "GENERALIZATION_COMPARISON.json")

    # 跨任务三子图合并为一张分组柱状图：left/right/overall × 三策略
    tasks = ["left", "right", "overall"]
    groups = [("VLA base", base), ("VLA few-shot", few), ("ACT", act)]
    x = np.arange(len(tasks)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=140)
    cols = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (name, d) in enumerate(groups):
        vals = [d["summary"][t]["success_rate"] for t in tasks]
        bars = ax.bar(x + (i-1)*w, vals, w, label=name, color=cols[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["push left", "push right", "overall"], fontsize=11)
    ax.set_ylabel("Success rate", fontsize=11)
    ax.set_title("Cross-task success: 5-shot adaptation vs ACT", fontsize=13)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(BASE / "cross_task_success_bar.png"); plt.close(fig)

    # 四维度横向对比：VLA vs ACT
    dims = ["seen task", "novel instr", "start pos", "target pos", "5-shot new task"]
    vla = [0.20, gc["vla"]["instruction"]["success_rate"],
           gc["vla"]["start_pos"]["success_rate"],
           gc["vla"]["target_pos"]["success_rate"],
           few["summary"]["overall"]["success_rate"]]
    actv = [gc["act"]["instruction"]["success_rate"],
            gc["act"]["instruction"]["success_rate"],
            gc["act"]["start_pos"]["success_rate"],
            gc["act"]["target_pos"]["success_rate"],
            act["summary"]["overall"]["success_rate"]]
    x = np.arange(len(dims)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=140)
    b1 = ax.bar(x - w/2, vla, w, label="VLA", color="#4C72B0")
    b2 = ax.bar(x + w/2, actv, w, label="ACT", color="#55A868")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(dims, fontsize=9, rotation=15)
    ax.set_ylabel("Success rate", fontsize=11)
    ax.set_title("VLA vs ACT across dimensions", fontsize=13)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(BASE / "dimension_comparison_bar.png"); plt.close(fig)

    print("plots -> cross_task_success_bar.png, dimension_comparison_bar.png")
    print("PLOT CROSS DONE")


if __name__ == "__main__":
    main()
