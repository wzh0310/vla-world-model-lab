#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""消融实验 消融与泛化实验可视化：统一风格高清图表。

输入:
  outputs/abl_h5|h10|h20|norew/train_summary.json  训练曲线
  outputs/abl_h5|h10|h20|norew/eval_standard.json  标准环境评估
  outputs/gen_imagine_v2/eval_generalize.json       泛化场景评估
  /data/wzh/robot/outputs/benchmarks/generalization/results.json  已有基准
输出 (results/):
  ablation_horizon.png        想象步长消融（训练探针曲线 + 真实指标柱状）
  ablation_reward.png         奖励预测器消融（想象奖励 + 真实指标对比）
  generalization_compare.png  泛化测试（想象策略 vs 已有算法基准）
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/wzh/robot/vla-world-model-lab"
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# 统一风格
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
GRAY = "#7f7f7f"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def train_summary(tag):
    return load(os.path.join(ROOT, f"outputs/abl_{tag}/train_summary.json"))


def eval_std(tag):
    return load(os.path.join(ROOT, f"outputs/abl_{tag}/eval_standard.json"))["summary"]


# ---------------------------------------------------------------- 图1：想象步长消融
horizons = [5, 10, 20]
tags_h = ["h5", "h10", "h20"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# 左：训练探针曲线（真实奖励头作为统一裁判）
ax = axes[0]
for h, tag, color in zip(horizons, tags_h, [BLUE, GREEN, ORANGE]):
    hist = train_summary(tag)["history"]
    probes = [(e["epoch"], e["probe_return"]) for e in hist if "probe_return" in e]
    if probes:
        xs, ys = zip(*probes)
        ax.plot(xs, ys, "-o", ms=3, color=color, label=f"H={h}")
ax.set_title("Imagination probe return by horizon")
ax.set_xlabel("Epoch")
ax.set_ylabel("Probe return (reward-head judge)")
ax.legend()

# 右：真实环境指标（成功率 + 平均奖励，双轴）
ax = axes[1]
evals_h = [eval_std(t) for t in tags_h]
succs = [e["success_rate"] * 100 for e in evals_h]
rewards = [e["mean_reward"] for e in evals_h]
x = np.arange(len(horizons))
w = 0.35
b1 = ax.bar(x - w/2, succs, w, color=BLUE, label="Success rate (%)")
ax2 = ax.twinx()
b2 = ax2.bar(x + w/2, rewards, w, color=RED, label="Mean reward")
ax.set_xticks(x); ax.set_xticklabels([f"H={h}" for h in horizons])
ax.set_ylabel("Success rate (%)", color=BLUE)
ax2.set_ylabel("Mean episode reward", color=RED)
ax.set_title("Real-env performance by horizon")
ax.set_ylim(-5, 105)
ax2.grid(False)
lines = [b1, b2]; ax.legend(lines, [l.get_label() for l in lines], loc="upper left")

fig.suptitle("Ablation 1: Imagination Horizon", fontsize=14)
fig.tight_layout()
out = os.path.join(RES, "ablation_horizon.png")
fig.savefig(out); plt.close(fig)
print("[PLOT]", out)

# ---------------------------------------------------------------- 图2：奖励预测器消融
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

s_wm = train_summary("h10")["summary"]      # reward=wm, H=10
s_zero = train_summary("norew")["summary"]  # reward=zero, H=10
e_wm = eval_std("h10")
e_zero = eval_std("norew")

labels = ["With reward predictor", "No reward (zero)"]
colors = [GREEN, RED]

# 想象奖励（训练信号）
ax = axes[0]
imag_rew = [s_wm["reward_late"], s_zero["reward_late"]]
bars = ax.bar(labels, imag_rew, color=colors)
ax.set_title("Imagined reward (training)")
ax.set_ylabel("Reward (normalized)")
for b, v in zip(bars, imag_rew):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom")

# 探针回报（统一奖励头裁判）
ax = axes[1]
probe = [s_wm["probe_late"], s_zero["probe_late"]]
bars = ax.bar(labels, probe, color=colors)
ax.set_title("Probe return (reward-head judge)")
ax.set_ylabel("Probe return")
for b, v in zip(bars, probe):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom")

# 真实环境成功率
ax = axes[2]
succ = [e_wm["success_rate"]*100, e_zero["success_rate"]*100]
bars = ax.bar(labels, succ, color=colors)
ax.set_title("Real-env success rate")
ax.set_ylabel("Success (%)")
ax.set_ylim(-5, 105)
for b, v in zip(bars, succ):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom")

fig.suptitle("Ablation 2: Reward Predictor", fontsize=14)
fig.tight_layout()
out = os.path.join(RES, "ablation_reward.png")
fig.savefig(out); plt.close(fig)
print("[PLOT]", out)

# ---------------------------------------------------------------- 图3：泛化测试对比
gen = load(os.path.join(ROOT, "outputs/gen_imagine_v2/eval_generalize.json"))["summary"]
bench = load("/data/wzh/robot/outputs/benchmarks/generalization/results.json")
# 取每个算法的 2cm 档做对比
bench_2cm = {}
for r in bench:
    if abs(float(r["start_range_cm"]) - 2.0) < 1e-6:
        bench_2cm[r["algorithm"]] = r["success_rate"]

names = list(bench_2cm.keys()) + ["WM-imagine"]
succs = [bench_2cm[n]*100 for n in bench_2cm] + [gen["success_rate"]*100]
cols = [GRAY]*len(bench_2cm) + [BLUE]

fig, ax = plt.subplots(figsize=(10, 4.6))
bars = ax.bar(names, succs, color=cols)
ax.set_title("Generalization: unseen cube start & target (2cm)")
ax.set_ylabel("Success rate (%)")
ax.set_ylim(-5, 105)
ax.tick_params(axis="x", rotation=20)
for b, v in zip(bars, succs):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
out = os.path.join(RES, "generalization_compare.png")
fig.savefig(out); plt.close(fig)
print("[PLOT]", out)

# ---------------------------------------------------------------- 汇总表
summary = {
    "horizon_ablation": [
        {"horizon": h, "train": train_summary(t)["summary"],
         "real_eval": eval_std(t)} for h, t in zip(horizons, tags_h)
    ],
    "reward_ablation": {
        "with_reward": {"train": s_wm, "real_eval": e_wm},
        "no_reward": {"train": s_zero, "real_eval": e_zero},
    },
    "generalization": gen,
}
with open(os.path.join(RES, "ablation_results.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("[PLOT] 汇总 ->", os.path.join(RES, "ablation_results.json"))
print("ABLATION_PLOT_OK")
