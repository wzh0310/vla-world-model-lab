#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多轨迹多步开环预测基准评估（用于调优对比，口径统一可复现）。

在固定的多条专家轨迹、固定起点集合上，做 N 步开环想象预测（仅首帧建后验，
全程不接真实图像），逐 step 计算 PSNR / SSIM，输出：
  - 每个 step 跨轨迹的均值与标准差（PSNR 曲线直接反映误差累积/模糊程度）
  - 聚合指标：整体均值、前3步、后3步、累积下降量
  - 可选：为指定轨迹生成 10 步「真实 vs 开环」对比 GIF / PNG

口径固定：同一 data、同一轨迹集合、同一起点、同一 steps，保证不同检查点
之间严格可比。视觉焦点：立方体位置是否清晰可辨（需人工看 GIF）。

用法（vla 环境）：
  python trainer/eval_multistep.py --ckpt outputs/wm_heads/checkpoints/wm_step004000.pt
  python trainer/eval_multistep.py --ckpt ... --vis-traj traj_0002 --out-dir outputs/wm_tune3
"""
import argparse
import glob
import json
import math
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import make_grid, save_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainer"))
from eval_single_step import build_models
from make_prediction_gif import load_one_trajectory, to_pil


def parse_args():
    ap = argparse.ArgumentParser(description="多轨迹多步开环预测基准评估")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/multistep_bench"))
    ap.add_argument("--trajs", default="traj_0002,traj_0007,traj_0011,traj_0015,traj_0020",
                    help="评估用专家轨迹（逗号分隔）")
    ap.add_argument("--starts", default="2,10,18", help="每条轨迹的起点集合（逗号分隔）")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--vis-traj", default="traj_0002", help="生成对比可视化用轨迹")
    ap.add_argument("--vis-start", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def psnr_of(pred, target):
    mse = F.mse_loss(pred.float(), target.float()).item()
    return 10.0 * math.log10(4.0 / (mse + 1e-8))


def ssim_of(pred, target):
    a = pred.float().clamp(-1, 1).cpu().numpy().transpose(1, 2, 0)
    b = target.float().clamp(-1, 1).cpu().numpy().transpose(1, 2, 0)
    return float(ssim(b, a, channel_axis=2, data_range=2.0))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ckpt_path = args.ckpt
    if ckpt_path is None:
        cands = sorted(glob.glob(os.path.join(ROOT, "outputs/wm_heads/checkpoints/wm_step*.pt")))
        ckpt_path = cands[-1]
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    M = build_models(args, ckpt)           # 复用：enc/dec/rssm/head（含 value_head 键安全跳过）
    enc, dec, rssm = M["enc"], M["dec"], M["rssm"]
    print(f"[BENCH] ckpt={ckpt_path} step={ckpt.get('step')} "
          f"stoch={M['stoch']} det={M['det']}", flush=True)

    trajs = [t.strip() for t in args.trajs.split(",") if t.strip()]
    starts = [int(s) for s in args.starts.split(",") if s.strip()]
    steps = args.steps
    per_step_psnr, per_step_ssim = [[] for _ in range(steps)], [[] for _ in range(steps)]

    with torch.no_grad(), h5py.File(args.data, "r") as f:
        avail = set(f.keys())
        trajs = [t for t in trajs if t in avail]
        for key in trajs:
            imgs, acts = load_one_trajectory(args.data, key, 64, args.device)
            for st in starts:
                if st + steps >= imgs.shape[0]:
                    continue
                e0 = enc(imgs[st].float().unsqueeze(0))[0]
                init = rssm.observe(rssm.initial_state(1, device=args.device),
                                    acts[st].unsqueeze(0), e0)[0]
                act_seq = acts[st + 1:st + 1 + steps].unsqueeze(0)
                states, _ = rssm.imagine_rollout(init, act_seq)
                preds = rssm.decode_rollout(states, dec)[:, 0]      # (steps,3,64,64)
                for k in range(steps):
                    real = imgs[st + 1 + k].float()
                    per_step_psnr[k].append(psnr_of(preds[k], real))
                    per_step_ssim[k].append(ssim_of(preds[k], real))
        n_eval = len(per_step_psnr[0]) if per_step_psnr and per_step_psnr[0] else 0

    psnr_mean = [float(np.mean(x)) for x in per_step_psnr]
    ssim_mean = [float(np.mean(x)) for x in per_step_ssim]
    psnr_std = [float(np.std(x)) for x in per_step_psnr]
    summary = {
        "ckpt": ckpt_path, "stoch": M["stoch"], "det": M["det"],
        "n_eval_rollouts": n_eval, "steps": steps, "trajs": trajs, "starts": starts,
        "psnr_mean_by_step": [round(x, 3) for x in psnr_mean],
        "psnr_std_by_step": [round(x, 3) for x in psnr_std],
        "ssim_mean_by_step": [round(x, 4) for x in ssim_mean],
        "psnr_overall": round(float(np.mean(psnr_mean)), 3),
        "psnr_first3": round(float(np.mean(psnr_mean[:3])), 3),
        "psnr_last3": round(float(np.mean(psnr_mean[-3:])), 3),
        "psnr_drop": round(float(np.mean(psnr_mean[:3]) - np.mean(psnr_mean[-3:])), 3),
        "ssim_last3": round(float(np.mean(ssim_mean[-3:])), 4),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "metrics"), exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics", "multistep_eval.json"), "w",
              encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[BENCH] 评估 {n_eval} 条开环轨迹 × {steps} 步", flush=True)
    print(f"[BENCH] PSNR by step: {[round(x,1) for x in psnr_mean]}", flush=True)
    print(f"[BENCH] 整体={summary['psnr_overall']}dB 前3={summary['psnr_first3']} "
          f"后3={summary['psnr_last3']} 下降={summary['psnr_drop']} "
          f"后3SSIM={summary['ssim_last3']}", flush=True)

    # 可选可视化：单条轨迹 10 步真实 vs 开环
    if args.vis_traj in avail:
        with torch.no_grad():
            imgs, acts = load_one_trajectory(args.data, args.vis_traj, 64, args.device)
            st = args.vis_start
            e0 = enc(imgs[st].float().unsqueeze(0))[0]
            init = rssm.observe(rssm.initial_state(1, device=args.device),
                                acts[st].unsqueeze(0), e0)[0]
            states, _ = rssm.imagine_rollout(init, acts[st + 1:st + 1 + steps].unsqueeze(0))
            preds = rssm.decode_rollout(states, dec)[:, 0]
            reals = imgs[st + 1:st + 1 + steps].float()
        vis_dir = os.path.join(args.out_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        grid = make_grid(torch.cat([reals, preds], dim=0), nrow=steps,
                         normalize=True, value_range=(-1, 1), pad_value=0.4)
        save_image(grid, os.path.join(vis_dir, "multistep_compare.png"))
        panel, bar = 192, 34
        gif_frames = []
        for k in range(steps):
            canvas = Image.new("RGB", (panel * 2 + 12, panel + bar), (32, 32, 38))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 10), f"t={k + 1:02d} REAL", fill=(235, 235, 235))
            draw.text((panel + 20, 10), f"OPEN-LOOP k={k + 1} PSNR={psnr_mean[k]:.1f}dB",
                      fill=(255, 214, 102))
            canvas.paste(to_pil(reals[k], panel), (0, bar))
            canvas.paste(to_pil(preds[k], panel), (panel + 12, bar))
            gif_frames.append(canvas)
        gif_frames[0].save(os.path.join(vis_dir, "multistep_compare.gif"),
                           save_all=True, append_images=gif_frames[1:],
                           duration=700, loop=0, disposal=2)
        print(f"[BENCH] 可视化: {vis_dir}/multistep_compare.{{png,gif}}", flush=True)
    print("[BENCH] OK", flush=True)


if __name__ == "__main__":
    main()
