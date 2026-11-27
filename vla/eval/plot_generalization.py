# -*- coding: utf-8 -*-
"""泛化对比柱状图：三类测试 × (VLA, ACT) 成功率。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("results/vla_generalization")
vla = json.load(open(BASE / "VLA_generalization_raw.json", encoding="utf-8"))
act = json.load(open(BASE / "ACT_generalization_raw.json", encoding="utf-8"))

keys = ["instruction", "start_pos", "target_pos", "overall"]
labels = ["Instruction\n(novel)", "Start pos\n(±5cm)", "Target pos\n(shift)", "Overall"]
v = [vla["summary"][k]["success_rate"] for k in keys]
a = [act["summary"][k]["success_rate"] for k in keys]

x = np.arange(len(keys)); w = 0.36
fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=140)
b1 = ax.bar(x - w/2, v, w, label="VLA (OpenVLA+LoRA)", color="#4C78A8")
b2 = ax.bar(x + w/2, a, w, label="ACT", color="#F58518")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Success rate", fontsize=11)
ax.set_title("Generalization success rate: VLA vs ACT (same scenes)", fontsize=13)
ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=10)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(BASE / "generalization_comparison_bar.png")
print("plot ->", BASE / "generalization_comparison_bar.png")
