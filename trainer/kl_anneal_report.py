# -*- coding: utf-8 -*-
"""KL 退火对照分析：从 TensorBoard events 读取基线（恒定 kl_scale）与退火训练的
KL/重建曲线，量化训练初期收敛稳定性，并（若已评估）对比最终预测精度。

产出 outputs/kl_anneal_report/：
  kl_anneal_compare.png   KL 曲线对比图（前 2000 步）
  kl_anneal_summary.json  稳定性统计（早期 KL 均值/方差/最小值）+ 精度对比

用法：python trainer/kl_anneal_report.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BASE_LOG = os.path.join(ROOT, "outputs/wm_tune1/logs")
ANNEAL_LOG = os.path.join(ROOT, "outputs/wm_klanneal/logs")
BASE_EVAL = os.path.join(ROOT, "outputs/eval_baseline/eval_metrics.json")
ANNEAL_EVAL = os.path.join(ROOT, "outputs/eval_klanneal/eval_metrics.json")
OUT = os.path.join(ROOT, "outputs/kl_anneal_report")


def read_scalars(log_dir, tag="loss/kl", max_steps=2000):
    """从 TensorBoard events 读取标量序列 -> (steps, values)。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(log_dir)
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        raise RuntimeError(f"{log_dir} 中无 {tag} 标量")
    ev = acc.Scalars(tag)
    steps = np.array([e.step for e in ev])
    vals = np.array([e.value for e in ev])
    mask = steps <= max_steps
    return steps[mask], vals[mask]


def stability(vals):
    """训练早期（前 500 记录点）KL 稳定性统计。"""
    early = vals[:500]
    return {"early_mean": float(np.mean(early)),
            "early_std": float(np.std(early)),
            "early_min": float(np.min(early)),
            "early_max": float(np.max(early)),
            "final": float(np.mean(vals[-50:]))}


def main():
    os.makedirs(OUT, exist_ok=True)
    s_base, v_base = read_scalars(BASE_LOG)
    s_ann, v_ann = read_scalars(ANNEAL_LOG)
    print(f"[KL] 基线 {len(v_base)} 点 | 退火 {len(v_ann)} 点", flush=True)

    st_base = stability(v_base)
    st_ann = stability(v_ann)

    # 对比图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(s_base, v_base, lw=1.6, color="#d62728",
            label=f"Baseline (fixed kl_scale=1.5), early std={st_base['early_std']:.3f}")
    ax.plot(s_ann, v_ann, lw=1.6, color="#1f77b4",
            label=f"KL annealing (0.1->1.5 over 1000 steps), early std={st_ann['early_std']:.3f}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL loss (with free bits)")
    ax.set_title("KL loss convergence: fixed weight vs annealing (first 2000 steps)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p_png = os.path.join(OUT, "kl_anneal_compare.png")
    fig.savefig(p_png, dpi=140)

    # 精度对比（若退火模型已评估）
    eval_cmp = None
    if os.path.exists(BASE_EVAL) and os.path.exists(ANNEAL_EVAL):
        b = json.load(open(BASE_EVAL, encoding="utf-8"))
        a = json.load(open(ANNEAL_EVAL, encoding="utf-8"))
        eval_cmp = {
            "baseline": {"step1_psnr": b["step1"]["psnr"], "step20_psnr": b["step20"]["psnr"],
                         "step1_ssim": b["step1"]["ssim"], "step20_ssim": b["step20"]["ssim"],
                         "psnr_drop": b["psnr_drop_1toN"],
                         "reward_mae_mean": b["reward_mae_mean"]},
            "kl_anneal": {"step1_psnr": a["step1"]["psnr"], "step20_psnr": a["step20"]["psnr"],
                          "step1_ssim": a["step1"]["ssim"], "step20_ssim": a["step20"]["ssim"],
                          "psnr_drop": a["psnr_drop_1toN"],
                          "reward_mae_mean": a["reward_mae_mean"]},
        }

    summary = {"baseline": st_base, "kl_anneal": st_ann, "eval_compare": eval_cmp}
    with open(os.path.join(OUT, "kl_anneal_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[KL] 早期稳定性  基线: mean={st_base['early_mean']:.4f} std={st_base['early_std']:.4f} "
          f"min={st_base['early_min']:.4f}", flush=True)
    print(f"[KL] 早期稳定性 退火: mean={st_ann['early_mean']:.4f} std={st_ann['early_std']:.4f} "
          f"min={st_ann['early_min']:.4f}", flush=True)
    if eval_cmp:
        print(f"[KL] step1 PSNR   基线={eval_cmp['baseline']['step1_psnr']:.2f} "
              f"退火={eval_cmp['kl_anneal']['step1_psnr']:.2f}", flush=True)
        print(f"[KL] step20 PSNR  基线={eval_cmp['baseline']['step20_psnr']:.2f} "
              f"退火={eval_cmp['kl_anneal']['step20_psnr']:.2f}", flush=True)
    print(f"[KL] 图: {p_png}", flush=True)


if __name__ == "__main__":
    main()
