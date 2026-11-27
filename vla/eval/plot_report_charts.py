# -*- coding: utf-8 -*-
"""生成实验报告统一风格图表（中文标注，数据全部来自已验证的真实结果）。

图表清单（输出到 results/vla_report_charts/）：
1. train_loss_curve.png   训练 loss 曲线（full_train_baseline_lr2e4 日志，滑动平均）
2. ablation_eval_loss.png LoRA rank 消融 best_eval_loss（r=4/8/16）
3. success_overview.png   四类实验成功率总览（消融对照/泛化/跨任务）

数据来源（只读，不修改）：
- results/vla_finetune_trainer/full_train_baseline_lr2e4.txt
- results/vla_ablation/summary.json
- results/vla_generalization/GENERALIZATION_COMPARISON.json
- results/vla_cross_task/vla_fewshot_raw.json / vla_baseline_raw.json / act_raw.json
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/wzh/robot/vla-world-model-lab")
OUT = ROOT / "results/vla_report_charts"
OUT.mkdir(parents=True, exist_ok=True)

# 中文字体：显式注册文泉驿正黑（matplotlib 缓存可能未包含系统字体）
from matplotlib import font_manager as fm
_FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if Path(_FONT).exists() and not any(f.name == "WenQuanYi Zen Hei" for f in fm.fontManager.ttflist):
    fm.fontManager.addfont(_FONT)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 150

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def parse_loss_log(path: Path):
    """从训练日志提取 (step, train_loss) 与 (step, eval_loss)。"""
    train, evals = [], []
    step = 0
    pat_train = re.compile(r"^\{'loss':\s*([\d.]+)")
    pat_eval = re.compile(r"^\{'eval_loss':\s*([\d.]+)")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            # 日志行常带进度条/空格前缀，截取第一个 '{' 之后的内容
            idx = line.find("{")
            line = line[idx:].strip() if idx >= 0 else ""
            m = pat_train.match(line)
            if m:
                step += 1
                train.append((step, float(m.group(1))))
                continue
            m = pat_eval.match(line)
            if m:
                evals.append((step, float(m.group(1))))
    return train, evals


def smooth(values, window=20):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        seg = values[lo:i + 1]
        out.append(sum(seg) / len(seg))
    return out


def chart_train_loss():
    log = ROOT / "results/vla_finetune_trainer/full_train_baseline_lr2e4.txt"
    if not log.exists():
        print("[skip] train log missing:", log)
        return
    train, evals = parse_loss_log(log)
    if not train:
        print("[skip] no train loss parsed")
        return
    xs = [s for s, _ in train]
    ys = [v for _, v in train]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, alpha=0.25, color=COLORS[0], lw=0.8, label="每步训练损失")
    ax.plot(xs, smooth(ys, 25), color=COLORS[0], lw=2,
            label="滑动平均（窗口 25）")
    if evals:
        ex = [s for s, _ in evals]
        ey = [v for _, v in evals]
        ax.plot(ex, ey, "o-", color=COLORS[1], lw=2, ms=5, label="验证损失（每 64 步）")
    ax.set_xlabel("优化器步数")
    ax.set_ylabel("损失")
    ax.set_title("QLoRA 训练损失曲线（rank=8, lr=2e-4, linear 调度）")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "train_loss_curve.png")
    plt.close(fig)
    print("[ok] train_loss_curve.png steps=%d eval_points=%d" % (len(train), len(evals)))


def chart_ablation():
    path = ROOT / "results/vla_ablation/summary.json"
    if not path.exists():
        print("[skip] ablation summary missing")
        return
    groups = json.load(open(path, encoding="utf-8"))["groups"]
    ranks = [g["rank"] for g in groups]
    evals_ = [g["best_eval_loss"] for g in groups]
    succ = [g["success_rate"] for g in groups]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    bars = ax1.bar([str(r) for r in ranks], evals_, color=COLORS[0], width=0.55)
    for b, v in zip(bars, evals_):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10)
    ax1.set_xlabel("LoRA 秩 r（alpha=2r）")
    ax1.set_ylabel("最优验证损失")
    ax1.set_title("拟合随秩提升：验证损失单调下降")
    ax1.set_ylim(0, max(evals_) * 1.18)
    ax1.grid(alpha=0.3, axis="y")

    bars2 = ax2.bar([str(r) for r in ranks], succ, color=COLORS[3], width=0.55)
    for b, v in zip(bars2, succ):
        ax2.text(b.get_x() + b.get_width() / 2, 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=10)
    ax2.set_xlabel("LoRA 秩 r")
    ax2.set_ylabel("闭环成功率")
    ax2.set_title("控制未改善：三组均 0/15")
    ax2.set_ylim(0, 0.15)
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("LoRA 秩消融（固定预算 192 步 / lr=1e-4 / seed=42）", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_eval_loss.png")
    plt.close(fig)
    print("[ok] ablation_eval_loss.png ranks=%s" % ranks)


def chart_success_overview():
    """汇总已验证的成功率数据；缺失文件则跳过对应条目。"""
    labels, values, colors = [], [], []
    # 消融对照（same budget）
    ab = ROOT / "results/vla_ablation/summary.json"
    if ab.exists():
        ref = json.load(open(ab, encoding="utf-8")).get("reference_not_same_budget", {})
        for key, lab in [("rank8_fulltrain", "rank8 全量训练"),
                         ("rank8_constlr", "rank8 恒定 lr")]:
            if key in ref:
                labels.append(lab)
                values.append(ref[key]["success_rate"])
                colors.append(COLORS[1])
    # 泛化
    gen = ROOT / "results/vla_generalization/GENERALIZATION_COMPARISON.json"
    if gen.exists():
        d = json.load(open(gen, encoding="utf-8"))
        # 结构为 {"vla": {...}, "act": {...}}，只取 VLA 一列
        s = d.get("vla", d.get("summary", d))
        for key, lab in [("instruction", "新指令泛化 VLA"),
                         ("start_pos", "起点泛化 VLA"),
                         ("target_pos", "目标泛化 VLA")]:
            node = s.get(key) if isinstance(s, dict) else None
            if isinstance(node, dict) and "success_rate" in node:
                labels.append(lab)
                values.append(node["success_rate"])
                colors.append(COLORS[0])
    # 跨任务 few-shot
    fs = ROOT / "results/vla_cross_task/vla_fewshot_raw.json"
    if fs.exists():
        d = json.load(open(fs, encoding="utf-8"))
        summary = d.get("summary", {})
        if isinstance(summary, dict) and "overall" in summary:
            node = summary["overall"]
        else:
            node = summary
        if isinstance(node, dict) and "success_rate" in node:
            labels.append("5-shot 跨任务 VLA")
            values.append(node["success_rate"])
            colors.append(COLORS[2])
    if not labels:
        print("[skip] no success data")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("闭环成功率")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("VLA 各实验闭环成功率总览（同一成功判据 <0.03m）")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "success_overview.png")
    plt.close(fig)
    print("[ok] success_overview.png items=%d" % len(labels))


if __name__ == "__main__":
    chart_train_loss()
    chart_ablation()
    chart_success_overview()
    print("CHARTS_DONE ->", OUT)
