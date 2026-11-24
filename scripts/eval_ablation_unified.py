#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一口径消融评估：用 WorldModel 推理封装评估多组检查点的多步预测。

所有模型走同一评估路径（WorldModel.from_checkpoint + rollout_from_trajectory），
兼容 RSSM 与 MLP 两种动力学，确保维度消融 / 动力学消融的口径完全一致。

评估集：测试集固定轨迹 + 固定起点（确定性），1/5/10/20 步关键指标：
  PSNR(dB)、SSIM、奖励 MAE（归一化尺度与原始 10x 尺度）。

产出 --out-dir：
  abl_unified_metrics.json   每组逐步长曲线 + 关键步汇总
  abl_unified_metrics.csv    关键步指标汇总表
  abl_unified_metrics.md     同上 Markdown

用法（vla 环境）：
  python scripts/eval_ablation_unified.py --groups-config results/abl_groups.json
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
from skimage.metrics import structural_similarity as ssim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.world_model import WorldModel

KEY_STEPS = (1, 5, 10, 20)


def parse_args():
    ap = argparse.ArgumentParser(description="统一口径消融评估")
    ap.add_argument("--groups-config", default=os.path.join(ROOT, "results/abl_groups.json"),
                    help="JSON: {显示名: checkpoint 路径}")
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/wm_dataset/test.h5"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--trajs", default="traj_0000,traj_0003,traj_0006,traj_0009,traj_0012,traj_0015,traj_0018")
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_traj(hdf5_path, key, image_size, device):
    with h5py.File(hdf5_path, "r") as f:
        low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
        high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
        span = torch.where(high == low, torch.ones_like(high), high - low)
        images = f[key]["images"][:]
        actions = f[key]["actions"][:]
        rewards = f[key]["rewards"][:]
    raw = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
    small = F.interpolate(raw, size=(image_size, image_size), mode="bilinear", align_corners=False)
    imgs = (small / 127.5 - 1.0)
    a = torch.as_tensor(actions, dtype=torch.float32, device=device)
    acts = ((a - low) / span * 2 - 1).clamp(-1, 1)
    r = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    rews = (r / 10.0).clamp(-1, 1)
    return imgs, acts, rews, r


def psnr(a, b):
    return 10.0 * math.log10(4.0 / (F.mse_loss(a.float(), b.float()).item() + 1e-8))


def ssim_of(a, b):
    x = a.detach().float().clamp(-1, 1).cpu().permute(1, 2, 0).numpy()
    y = b.detach().float().clamp(-1, 1).cpu().permute(1, 2, 0).numpy()
    return float(ssim(y, x, channel_axis=2, data_range=2.0))


import math


def evaluate_one(wm, args, trajs):
    S = wm.image_size
    acc = {"psnr": [[] for _ in range(args.steps)],
           "ssim": [[] for _ in range(args.steps)],
           "rew_norm": [[] for _ in range(args.steps)],
           "rew_raw": [[] for _ in range(args.steps)]}
    n = 0
    with torch.no_grad(), h5py.File(args.data, "r") as f:
        avail = set(f.keys())
        for key in trajs:
            if key not in avail:
                continue
            imgs, acts, rews, raws = load_traj(args.data, key, S, args.device)
            st = args.start
            if st + args.warmup + args.steps > imgs.shape[0]:
                continue
            out = wm.rollout_from_trajectory(imgs, acts, warmup_steps=args.warmup,
                                             horizon=args.steps, start=st, sample=False)
            preds = out["images"]
            pred_r = out["rewards"]
            # 真实帧对齐：从 warmup 之后开始
            real_start = st + args.warmup
            for k in range(args.steps):
                real_img = imgs[real_start + k]
                acc["psnr"][k].append(psnr(preds[k], real_img))
                acc["ssim"][k].append(ssim_of(preds[k], real_img))
                if pred_r is not None:
                    # 奖励按「到达帧」索引，与图像帧对齐：pred[k] 对应 real_start+k
                    j = real_start + k
                    j = min(j, len(rews) - 1)
                    acc["rew_norm"][k].append(pred_r[k].float().item() - rews[j].item())
                    acc["rew_raw"][k].append(pred_r[k].float().item() * 10.0 - raws[j].item())
            n += 1
    mean = {k: [float(np.mean(x)) if x else 0.0 for x in v] for k, v in acc.items()}
    return n, mean


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.groups_config, encoding="utf-8") as f:
        groups = json.load(f)
    trajs = [t.strip() for t in args.trajs.split(",") if t.strip()]
    results = {}
    for name, ckpt in groups.items():
        if not os.path.isfile(ckpt):
            print(f"[EVAL] 跳过 {name}: 检查点不存在 {ckpt}", flush=True)
            continue
        wm = WorldModel.from_checkpoint(ckpt, device=args.device)
        n, mean = evaluate_one(wm, args, trajs)
        key_steps = {}
        for k in KEY_STEPS:
            if k <= args.steps:
                key_steps[f"step{k}"] = {
                    "psnr_db": round(mean["psnr"][k - 1], 3),
                    "ssim": round(mean["ssim"][k - 1], 4),
                    "reward_mae_norm": round(abs(mean["rew_norm"][k - 1]), 4),
                    "reward_mae_raw": round(abs(mean["rew_raw"][k - 1]), 4),
                }
        results[name] = {
            "ckpt": ckpt, "dynamics": wm.dynamics, "stoch": wm.stoch_dim,
            "n_traj": n, "steps": args.steps,
            "psnr_by_step": [round(x, 3) for x in mean["psnr"]],
            "ssim_by_step": [round(x, 4) for x in mean["ssim"]],
            "reward_mae_norm_by_step": [round(abs(x), 4) for x in mean["rew_norm"]],
            "reward_mae_raw_by_step": [round(abs(x), 4) for x in mean["rew_raw"]],
            "psnr_step1": round(mean["psnr"][0], 3),
            "psnr_step20": round(mean["psnr"][-1], 3),
            "psnr_drop_1to20": round(mean["psnr"][0] - mean["psnr"][-1], 3),
            "key_steps": key_steps,
        }
        print(f"[EVAL] {name}: dyn={wm.dynamics} stoch={wm.stoch_dim} n={n} "
              f"PSNR@20={results[name]['psnr_step20']:.2f} SSIM@20={mean['ssim'][-1]:.4f}", flush=True)

    with open(os.path.join(args.out_dir, "abl_unified_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    header = ["模型", "动力学", "stoch", "PSNR@1", "PSNR@5", "PSNR@10", "PSNR@20",
              "SSIM@5", "SSIM@10", "SSIM@20", "奖励MAE@20(raw)", "PSNR衰减(1->20)"]
    rows = []
    for name, r in results.items():
        ks = r["key_steps"]
        rows.append([name, r["dynamics"], r["stoch"],
                     ks.get("step1", {}).get("psnr_db"), ks.get("step5", {}).get("psnr_db"),
                     ks.get("step10", {}).get("psnr_db"), ks.get("step20", {}).get("psnr_db"),
                     ks.get("step5", {}).get("ssim"), ks.get("step10", {}).get("ssim"),
                     ks.get("step20", {}).get("ssim"), ks.get("step20", {}).get("reward_mae_raw"),
                     r["psnr_drop_1to20"]])
    with open(os.path.join(args.out_dir, "abl_unified_metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(header)
        for row in rows:
            w.writerow(["" if v is None else round(v, 4) if isinstance(v, float) else v for v in row])
    with open(os.path.join(args.out_dir, "abl_unified_metrics.md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(map(str, header)) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join("-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in row) + " |\n")
    print(f"[EVAL] 汇总已写入 {args.out_dir}/abl_unified_metrics.*", flush=True)


if __name__ == "__main__":
    main()
