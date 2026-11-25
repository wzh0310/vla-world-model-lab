#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""绘制 VLA 闭环推理测试结果：EE 轨迹俯视图 + 立方体距离曲线 + 动作方向对比。"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(ROOT, "results", "vla_closed_loop")
OUT = os.path.join(BASE, "closed_loop_analysis.png")

COLORS = ["#1976D2", "#E65100", "#388E3C"]


def load():
    s = json.load(open(os.path.join(BASE, "summary.json"), encoding="utf-8"))
    eps = []
    for i in range(len(s["episodes"])):
        eps.append(json.load(open(os.path.join(BASE, f"ep{i:02d}", "episode.json"),
                                  encoding="utf-8")))
    return s, eps


def main():
    s, eps = load()
    short = ["push to red target", "move to red circle", "push onto red goal"]
    labels = [f"ep{i}: {short[i]}" if i < len(short) else f"ep{i}"
              for i, e in enumerate(eps)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # (1) EE 轨迹俯视图（相对起点累计位移）
    ax = axes[0]
    for i, e in enumerate(eps):
        ee = np.array([r["ee_delta"][:2] for r in e["steps"]])
        traj = np.vstack([[0, 0], np.cumsum(ee, axis=0)])
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=3, color=COLORS[i], label=labels[i], alpha=0.85)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=70, color=COLORS[i], marker="*", zorder=5)
    ax.axhline(0, color="gray", lw=0.6, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.6, alpha=0.5)
    ax.set_xlabel("EE cumulative dx (m)")
    ax.set_ylabel("EE cumulative dy (m)")
    ax.set_title("EE trajectory (top view, relative to start)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)

    # (2) 立方体到目标距离曲线
    ax = axes[1]
    for i, e in enumerate(eps):
        d = [r["dist_to_target"] for r in e["steps"]]
        ax.plot(range(len(d)), d, "-", color=COLORS[i], label=labels[i], lw=2, alpha=0.85)
    ax.axhline(0.03, color="red", ls="--", lw=1, label="success radius 0.03m")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cube-target distance (m)")
    ax.set_title("Cube distance to target (no progress expected)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # (3) 三指令平均动作方向（xy）
    ax = axes[2]
    means = np.array([[np.mean([r["mapped_4d"][0] for r in e["steps"]]),
                       np.mean([r["mapped_4d"][1] for r in e["steps"]])] for e in eps])
    for i in range(len(eps)):
        mx, my = means[i, 0] * 1000, means[i, 1] * 1000
        ax.scatter([mx], [my], s=1, color=COLORS[i])  # 计入数据范围
        ax.annotate("", xy=(mx, my), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS[i], lw=2.5,
                                    mutation_scale=18, shrinkA=0, shrinkB=0))
    for i in range(len(eps)):
        ax.plot([], [], color=COLORS[i], lw=2.5, label=labels[i])
    lim = max(6.0, float(np.abs(means * 1000).max()) * 1.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", lw=0.6, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.6, alpha=0.5)
    ax.set_xlabel("mean action dx (mm/step)")
    ax.set_ylabel("mean action dy (mm/step)")
    ax.set_title("Mean mapped action direction by instruction")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    fig.suptitle(f"VLA closed-loop rollout | success={s['success_rate']} | "
                 f"mean latency={s['mean_latency_s']}s/step | {s['config']['max_steps']} steps/episode",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("[PLOT] saved", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
