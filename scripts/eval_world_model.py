#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""世界模型统一评估入口：输入检查点 + 测试轨迹，自动输出全部指标与「精度-步长」衰减曲线。

指标体系（trainer/metrics.py）：
  像素级   MSE / PSNR（逐 step）
  结构级   SSIM（逐 step）
  感知级   ResNet18 特征感知距离（逐 step；LPIPS 替代实现，见 metrics.py 说明）
  动力学级 隐状态每步漂移（开环特征 vs teacher-forcing 后验特征，归一化 L2）
           奖励预测 MAE（逐 step，归一化尺度）

口径：每条 (轨迹, 起点) 先用首帧后验作起点（warmup=1，与 eval_multistep_full 一致），
再用真实动作序列开环想象 N 步；teacher-forcing 分支用同一动作序列逐帧 observe，
两者特征差即动力学漂移。像素/结构/感知指标按「到达帧」对齐真实帧。

产出（--out-dir）：
  eval_curves.png      精度-步长衰减曲线（4 子图：PSNR/SSIM/感知距离/隐状态漂移）
  eval_metrics.json    逐 step 指标均值 + 汇总
  eval_metrics.csv     逐 step 表格（便于对比不同模型）

用法（vla 环境）：
  python scripts/eval_world_model.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt \
      --trajs traj_0002,traj_0007,traj_0011,traj_0015,traj_0020 --starts 2,8 --steps 20
"""
import argparse
import csv
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainer"))
from models.world_model import WorldModel
from trainer.metrics import (PerceptualMetric, latent_drift_l2, mse_metric,
                             psnr_metric, reward_mae, ssim_metric)


def parse_args():
    ap = argparse.ArgumentParser(description="世界模型统一评估（全部指标 + 衰减曲线）")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/wm_tune1/checkpoints/wm_step004000.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/eval_baseline"))
    ap.add_argument("--trajs", default="traj_0002,traj_0007,traj_0011,traj_0015,traj_0020")
    ap.add_argument("--starts", default="2,8", help="每条轨迹的起点索引，逗号分隔")
    ap.add_argument("--steps", type=int, default=20, help="开环预测步数")
    ap.add_argument("--no-perceptual", action="store_true", help="跳过感知级指标（首次加载较慢）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_all(data_path, trajs):
    """一次性读入全部轨迹：图像 224->64 [-1,1]、动作/奖励归一化（与训练口径一致）。"""
    store = {}
    with h5py.File(data_path, "r") as f:
        a_low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32))
        a_high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32))
        span = torch.where(a_high == a_low, torch.ones_like(a_high), a_high - a_low)
        for key in trajs:
            raw = torch.as_tensor(f[key]["images"][:]).permute(0, 3, 1, 2).float()
            small = F.interpolate(raw, size=(64, 64), mode="bilinear", align_corners=False)
            imgs = small / 127.5 - 1.0
            acts = ((torch.as_tensor(f[key]["actions"][:], dtype=torch.float32) - a_low)
                    / span * 2 - 1).clamp(-1, 1)
            rews = (torch.as_tensor(f[key]["rewards"][:], dtype=torch.float32) / 10.0).clamp(-1, 1)
            store[key] = {"images": imgs, "actions": acts, "rewards": rews}
    return store


@torch.no_grad()
def eval_one(wm, ep, start, steps, perceptual=None):
    """单条 (轨迹, 起点) 的开环评估，返回逐 step 指标 dict。"""
    imgs, acts, rews = ep["images"], ep["actions"], ep["rewards"]
    dev = wm.device
    N = steps
    if start + 1 + N > imgs.shape[0]:
        raise ValueError(f"start={start} + steps={N} 超出轨迹长度 {imgs.shape[0]}")

    # teacher-forcing 分支：真实帧逐帧 observe（得到每步后验特征，漂移参照）
    state = wm.rssm.initial_state(1, device=dev)
    f_tf = []
    for t in range(start, start + 1 + N):
        state, _ = wm.rssm.observe(state, acts[t].to(dev), wm._encode(imgs[t].to(dev)))
        f_tf.append(wm.rssm.get_feature(state).reshape(-1))
    f_tf = torch.stack(f_tf, 0)                                    # (N+1,F)，第 0 项为起点

    # 开环分支：首帧后验作起点 + 真实动作开环想象
    init_state = wm.rssm.initial_state(1, device=dev)
    init_state, _ = wm.rssm.observe(init_state, acts[start].to(dev),
                                    wm._encode(imgs[start].to(dev)))
    acts_seq = acts[start + 1: start + 1 + N].unsqueeze(0).to(dev)
    states, _ = wm.rssm.imagine_rollout(init_state, acts_seq, sample=True)
    f_ol = torch.stack([wm.rssm.get_feature(s).reshape(-1) for s in states], 0)  # (N,F)

    preds = wm.rssm.decode_rollout(states, wm.decoder).clamp(-1, 1)[:, 0].cpu()  # (N,3,64,64)
    reals = imgs[start + 1: start + 1 + N]                                       # (N,3,64,64)

    m = {
        "mse": mse_metric(preds, reals),
        "psnr": psnr_metric(preds, reals),
        "ssim": ssim_metric(preds, reals),
        "latent_drift": latent_drift_l2(f_ol, f_tf[1:]),
        "reward_err": (pred_rewards - rews[start + 1: start + 1 + N]).abs().numpy()
        if (pred_rewards := (wm.reward_head(f_ol.to(dev)).squeeze(-1).cpu())) is not None else None,
    }
    if perceptual is not None:
        m["perceptual"] = perceptual.distance(preds, reals)
    return m


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    wm = WorldModel.from_checkpoint(args.ckpt, device=args.device)
    trajs = [t.strip() for t in args.trajs.split(",") if t.strip()]
    starts = [int(s) for s in args.starts.split(",") if s.strip()]
    perceptual = None if args.no_perceptual else PerceptualMetric(torch.device(args.device))
    if perceptual is None:
        print("[EVAL] 已跳过感知级指标（--no-perceptual）", flush=True)

    store = load_all(args.data, trajs)
    # 逐 step 累积（所有 轨迹×起点 合并求均值）
    acc = {k: [] for k in ["mse", "psnr", "ssim", "latent_drift", "reward_err", "perceptual"]}
    n_roll = 0
    for key in trajs:
        ep = store[key]
        for st in starts:
            m = eval_one(wm, ep, st, args.steps, perceptual)
            for k, v in m.items():
                if v is not None:
                    acc[k].append(v)
            n_roll += 1
            print(f"[EVAL] {key} start={st} 完成（{n_roll}/{len(trajs) * len(starts)}）", flush=True)

    # 合并为逐 step 均值曲线
    curves = {}
    for k, lst in acc.items():
        if lst:
            curves[k] = np.mean(np.stack(lst, 0), axis=0).tolist()   # (steps,)

    # 衰减曲线图（4 子图）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(1, args.steps + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("psnr", "PSNR (dB)", "Pixel-level accuracy", "#d62728"),
        ("ssim", "SSIM", "Structural similarity", "#1f77b4"),
        ("perceptual", "Perceptual distance", "ResNet18-feature cosine (lower=better)", "#9467bd"),
        ("latent_drift", "Latent drift (L2/sqrt(F))", "Open-loop vs teacher-forcing hidden state", "#2ca02c"),
    ]
    for ax, (key, ylabel, title, color) in zip(axes.flat, panels):
        if key not in curves:
            ax.set_visible(False)
            continue
        ax.plot(xs, curves[key], marker="o", ms=3.5, lw=2, color=color)
        ax.set_xlabel("Open-loop step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Prediction accuracy vs horizon ({len(trajs)} trajs x {len(starts)} starts x {args.steps} steps)",
                 fontsize=12)
    fig.tight_layout()
    curves_path = os.path.join(args.out_dir, "eval_curves.png")
    fig.savefig(curves_path, dpi=140)
    plt.close(fig)

    # CSV：逐 step 表格
    csv_path = os.path.join(args.out_dir, "eval_metrics.csv")
    cols = ["step"] + [k for k in ["mse", "psnr", "ssim", "perceptual", "latent_drift", "reward_err"]
                       if k in curves]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(args.steps):
            w.writerow([i + 1] + [round(curves[c][i], 5) for c in cols[1:]])

    # JSON 汇总
    summary = {
        "ckpt": args.ckpt,
        "trajs": trajs, "starts": starts, "steps": args.steps,
        "n_rollouts": n_roll,
        "step1": {k: curves[k][0] for k in curves},
        f"step{args.steps}": {k: curves[k][-1] for k in curves},
        "psnr_drop_1toN": curves["psnr"][0] - curves["psnr"][-1],
        "reward_mae_mean": float(np.mean(curves["reward_err"])) if "reward_err" in curves else None,
        "curves_by_step": {k: [round(x, 5) for x in v] for k, v in curves.items()},
    }
    json_path = os.path.join(args.out_dir, "eval_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[EVAL] ===== 基线量化得分（{n_roll} 条开环 x {args.steps} 步）=====", flush=True)
    for k in ["psnr", "ssim", "perceptual", "latent_drift"]:
        if k in curves:
            print(f"  {k:14s} step1={curves[k][0]:8.4f}  step{args.steps}={curves[k][-1]:8.4f}", flush=True)
    if "reward_err" in curves:
        print(f"  reward_mae     mean={summary['reward_mae_mean']:.5f}", flush=True)
    print(f"[EVAL] 衰减曲线: {curves_path}", flush=True)
    print(f"[EVAL] 指标文件: {json_path} / {csv_path}", flush=True)


if __name__ == "__main__":
    main()
