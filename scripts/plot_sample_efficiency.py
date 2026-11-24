#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""绘制样本效率对比曲线 + 计算样本节省比例。

输入: results/sfe_results.json（含 PPO 四档 + WM 四档的统一评估指标）
输出:
  results/sample_efficiency_curves.png  数据量-性能对比曲线（累计奖励 + 最终距离 + 成功率）
  results/sample_efficiency_summary.json  样本节省比例计算结果
横轴统一为"训练数据量比例"（PPO 用 checkpoint 步数/200万，WM 用轨迹比例）。
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PCTS = [10, 30, 50, 100]
XLABEL = "Training data (%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/sfe_results.json")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = json.load(f)

    # 按 group + pct 归集
    def level_to_pct(r):
        if r["group"] == "ppo":
            return round(r["data_level"] / 2_000_000 * 100)
        return r["data_level"]

    data = {"ppo": {}, "wm": {}}
    for r in rows:
        data[r["group"]][level_to_pct(r)] = r

    def series(group, key):
        return [data[group].get(p, {}).get(key, np.nan) for p in PCTS]

    ppo_rew = series("ppo", "mean_reward")
    wm_rew = series("wm", "mean_reward")
    ppo_dist = series("ppo", "mean_final_dist")
    wm_dist = series("wm", "mean_final_dist")
    ppo_succ = series("ppo", "success_rate")
    wm_succ = series("wm", "success_rate")

    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = {"ppo": "#d62728", "wm": "#1f77b4"}

    # 累计奖励（越高越好，越接近0越好）
    ax = axes[0]
    ax.plot(PCTS, ppo_rew, "-o", color=colors["ppo"], label="Visual PPO (online)")
    ax.plot(PCTS, wm_rew, "-s", color=colors["wm"], label="World Model + Imagination")
    ax.set_title("Cumulative Reward (higher is better)")
    ax.set_xlabel(XLABEL); ax.set_ylabel("Episode reward")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 最终距离（越低越好）
    ax = axes[1]
    ax.plot(PCTS, ppo_dist, "-o", color=colors["ppo"], label="Visual PPO (online)")
    ax.plot(PCTS, wm_dist, "-s", color=colors["wm"], label="World Model + Imagination")
    ax.set_title("Final Distance to Target (lower is better)")
    ax.set_xlabel(XLABEL); ax.set_ylabel("Distance (m)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 成功率
    ax = axes[2]
    ax.plot(PCTS, np.array(ppo_succ) * 100, "-o", color=colors["ppo"], label="Visual PPO (online)")
    ax.plot(PCTS, np.array(wm_succ) * 100, "-s", color=colors["wm"], label="World Model + Imagination")
    ax.set_title("Success Rate")
    ax.set_xlabel(XLABEL); ax.set_ylabel("Success (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_ylim(-5, 105)

    fig.suptitle("Sample Efficiency: World Model vs Visual PPO on PushCube", fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "sample_efficiency_curves.png")
    fig.savefig(out_png, dpi=150)
    print(f"[SFE] 曲线图 -> {out_png}")

    # 样本节省比例：诚实判定。
    # 只有「两组在真实环境都达到非零成功率」时，基于成功率的样本节省才有统计意义。
    # 若所有档位成功率都饱和在 0（本实验的真实情况），任何基于连续指标的"达标"都是
    # 失败水平之间的比较，属于假阳性，必须判定为无意义，不得包装成正面结论。
    ppo_full_dist = data["ppo"].get(100, {}).get("mean_final_dist", np.nan)
    ppo_full_succ = data["ppo"].get(100, {}).get("success_rate", 0.0)
    wm_succs = np.array(series("ppo" if False else "wm", "success_rate"), dtype=float)
    any_success = (ppo_full_succ > 0) or (np.nanmax(wm_succs) > 0 if np.isfinite(wm_succs).any() else False)

    save_ratio = None
    saving_basis = None
    if any_success:
        # 基于成功率计算：找第一个成功率 >= PPO满数据成功率的 WM 档位
        if ppo_full_succ > 0:
            reached = [p for p, s in zip(PCTS, wm_succs)
                       if np.isfinite(s) and s >= ppo_full_succ]
            if reached:
                wm_needed = min(reached)
                save_ratio = round(100.0 / wm_needed, 2)
                saving_basis = "success_rate"
    saturated = not any_success
    summary = {
        "metric_used_for_saving": "success_rate (only meaningful if non-saturated)",
        "performance_saturated_at_zero": saturated,
        "ppo_full_data_dist": ppo_full_dist,
        "ppo_full_data_reward": data["ppo"].get(100, {}).get("mean_reward"),
        "ppo_full_data_success": ppo_full_succ,
        "wm_full_data_dist": data["wm"].get(100, {}).get("mean_final_dist"),
        "wm_full_data_reward": data["wm"].get(100, {}).get("mean_reward"),
        "wm_full_data_success": data["wm"].get(100, {}).get("success_rate"),
        "sample_saving_factor": save_ratio,
        "saving_basis": saving_basis,
        "note": ("样本节省倍数仅在两组真实成功率非饱和时有统计意义。本实验中纯视觉 PPO "
                 "与世界模型想象策略在真实环境成功率均为 0%（性能饱和在失败水平），"
                 "故样本节省倍数判定为 None——不能把失败水平之间的连续指标差异包装成"
                 "样本效率优势。曲线的价值在于揭示：PPO 训练后期策略坍缩（600k 步奖励"
                 "-0.52/距离0.09 反而优于 2000k 步的 -2.49/0.20），世界模型在想象世界"
                 "内有清晰学习曲线（探针回报随数据/训练提升），但 sim-to-real 间隙仍导致"
                 "真实环境零样本迁移失败。"),
        "all_series": {
            "data_pct": PCTS,
            "ppo_reward": ppo_rew, "wm_reward": wm_rew,
            "ppo_dist": ppo_dist, "wm_dist": wm_dist,
            "ppo_success": ppo_succ, "wm_success": wm_succ,
        },
    }
    out_json = os.path.join(args.out_dir, "sample_efficiency_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SFE] 汇总 -> {out_json}")
    print(f"[SFE] 样本节省倍数 = {save_ratio}")


if __name__ == "__main__":
    main()
