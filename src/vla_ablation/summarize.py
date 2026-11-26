#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总固定预算 LoRA 秩消融（r=4/8/16）结果：对比表 + 柱状图 + JSON。

读取 results/vla_ablation/rank{r}/ 下的 trainer_train_result.json、best_checkpoint.json、
eval.json；输出 comparison_table.md、summary.json、三张 PNG。
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "vla_ablation"
RANKS = [4, 8, 16]


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []
    for r in RANKS:
        d = BASE / f"rank{r}"
        tr = load_json(d / "trainer_train_result.json")
        bc = load_json(d / "best_checkpoint.json")
        ev = load_json(d / "eval.json")
        rows.append({
            "rank": r,
            "alpha": tr["config"]["lora_alpha"],
            "trainable_params": tr["trainable_params"],
            "trainable_pct": round(tr["trainable_params"] / (tr["n_params_b"] * 1e9) * 100, 4),
            "steps": tr["steps"],
            "loss_first": tr["loss_first"],
            "loss_last": tr["loss_last"],
            "best_eval_loss": bc["eval_loss"],
            "best_step": bc["global_step"],
            "final_eval_loss": round(tr["eval_loss"], 6),
            "total_episodes": ev["total_episodes"],
            "total_successes": ev["total_successes"],
            "success_rate": ev["success_rate"],
            "mean_dist_improvement": ev["mean_dist_improvement"],
            "per_instruction": ev["per_instruction"],
            "elapsed_s": tr["elapsed_s"],
        })

    # 参考基线（非同一预算，仅作对照说明，不混入柱状图）
    refs = {}
    zeroshot = ROOT / "results" / "vla_eval" / "zeroshot.json"
    finetuned = ROOT / "results" / "vla_eval" / "finetuned.json"
    if zeroshot.exists():
        z = load_json(zeroshot)
        refs["zero_shot_full"] = {"success_rate": z["success_rate"],
                                  "mean_dist_improvement": z["mean_dist_improvement"]}
    if finetuned.exists():
        f = load_json(finetuned)
        refs["rank8_fulltrain"] = {"success_rate": f["success_rate"],
                                   "mean_dist_improvement": f["mean_dist_improvement"]}
    constlr = BASE / "rank8_constlr" / "eval.json"
    if constlr.exists():
        c = load_json(constlr)
        refs["rank8_constlr"] = {"success_rate": c["success_rate"],
                                 "mean_dist_improvement": c["mean_dist_improvement"]}

    summary = {"protocol": {
        "fixed": ["dataset(hf_dataset v1.0)", "lr=1e-4", "warmup=50", "seed=42",
                  "max_steps=192", "eval: 5 instr x 3 episodes, seed_base=100, max_steps=40"],
        "varied": "LoRA rank/alpha only (alpha/r=2 constant)"},
        "groups": rows, "reference_not_same_budget": refs}
    with open(BASE / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------- 对比表 ----------
    lines = [
        "# LoRA 秩消融实验结果（固定 192 步预算）",
        "",
        "同一数据集、同一学习率（1e-4）、同一 seed（42）、同一 192 步训练预算；",
        "仅改变 LoRA 秩 r 与 alpha（保持 alpha/r=2，隔离缩放因子影响）。",
        "闭环评估统一使用 `vla/eval/eval.py`：5 条标准指令 × 3 回合，最大 40 步，",
        "成功判据为方块中心距红盘 < 0.03 m。",
        "",
        "| LoRA 秩 r | alpha | 可训练参数 | 占底座比例 | loss 首→末 | 最优 eval_loss (step) | 成功率 | 平均距离改善 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for x in rows:
        lines.append(
            f"| r={x['rank']} | {x['alpha']} | {x['trainable_params']:,} | {x['trainable_pct']}% | "
            f"{x['loss_first']} → {x['loss_last']} | {x['best_eval_loss']} (step {x['best_step']}) | "
            f"{x['total_successes']}/{x['total_episodes']} = {x['success_rate']} | "
            f"{x['mean_dist_improvement']} m |")
    lines += ["",
              "## 对照组（不同训练预算，仅作定性参照，不参与主表结论）",
              ""]
    if refs:
        lines.append("| 对照 | 成功率 | 平均距离改善 | 说明 |")
        lines.append("| --- | --- | --- | --- |")
        if "zero_shot_full" in refs:
            z = refs["zero_shot_full"]
            lines.append(f"| 零样本底座 int4 | {z['success_rate']} | {z['mean_dist_improvement']} m | 未微调 |")
        if "rank8_fulltrain" in refs:
            z = refs["rank8_fulltrain"]
            lines.append(f"| r=8 全量训练最优权重 | {z['success_rate']} | {z['mean_dist_improvement']} m | "
                         "lr schedule/步数与本消融不同 |")
        if "rank8_constlr" in refs:
            z = refs["rank8_constlr"]
            lines.append(f"| r=8 constant-lr（同 192 步） | {z['success_rate']} | {z['mean_dist_improvement']} m | "
                         "仅把 linear 改为 constant，验证 lr 退火影响 |")
    lines += ["",
              "## 分指令成功率（本消融）", "",
              "| 指令 | r=4 | r=8 | r=16 |", "| --- | --- | --- | --- |"]
    instrs = list(rows[0]["per_instruction"].keys())
    for ins in instrs:
        cells = [f"{x['per_instruction'][ins]['successes']}/{x['per_instruction'][ins]['episodes']}"
                 for x in rows]
        lines.append(f"| {ins} | " + " | ".join(cells) + " |")
    (BASE / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- 柱状图 ----------
    labels = [f"r={x['rank']}" for x in rows]
    xpos = np.arange(len(labels))
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    def barfig(vals, ylabel, title, path, fmt="{:.4f}", ylim=None):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
        bars = ax.bar(xpos, vals, color=colors, width=0.58)
        ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11); ax.set_title(title, fontsize=13)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=10)
        fig.tight_layout(); fig.savefig(path); plt.close(fig)

    # 成功率图：三组秩 + constant-lr 对照（验证 scheduler 影响）
    sr_vals = [x["success_rate"] for x in rows]
    sr_labels = labels[:]
    sr_colors = colors[:]
    if "rank8_constlr" in refs:
        sr_vals.append(refs["rank8_constlr"]["success_rate"])
        sr_labels.append("r=8\nconst-lr")
        sr_colors.append("#E377C2")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    xpos2 = np.arange(len(sr_vals))
    bars = ax.bar(xpos2, sr_vals, color=sr_colors, width=0.58)
    ax.set_xticks(xpos2); ax.set_xticklabels(sr_labels, fontsize=10)
    ax.set_ylabel("Success rate", fontsize=11)
    ax.set_title("Closed-loop success rate (192-step budget)", fontsize=13)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, sr_vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout(); fig.savefig(BASE / "success_rate_bar.png"); plt.close(fig)
    barfig([x["mean_dist_improvement"] for x in rows], "Distance improvement (m)",
           "Mean distance improvement vs LoRA rank", BASE / "dist_improvement_bar.png",
           "{:.4f}")
    barfig([x["best_eval_loss"] for x in rows], "Best eval loss",
           "Best eval loss vs LoRA rank (192-step budget)", BASE / "eval_loss_bar.png",
           "{:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    bars = ax.bar(xpos, [x["trainable_params"] / 1e6 for x in rows],
                  color=colors, width=0.58)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Trainable params (M)", fontsize=11)
    ax.set_title("LoRA trainable parameters vs rank", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, [x["trainable_params"] / 1e6 for x in rows]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}M", ha="center", va="bottom", fontsize=10)
    fig.tight_layout(); fig.savefig(BASE / "trainable_params_bar.png"); plt.close(fig)

    print("[summary] ->", BASE)
    for x in rows:
        print(f"  r={x['rank']}: success={x['success_rate']} "
              f"dist_imp={x['mean_dist_improvement']} best_eval={x['best_eval_loss']}")


if __name__ == "__main__":
    main()
