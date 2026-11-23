#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成连续 10 步「真实帧 - 单步预测帧」对比 GIF（第 9 周 Day7）。

单步预测定义：以真实历史帧建立的后验状态为起点，用 imagine_step(a_t)
纯动力学预测下一帧 x_t（不使用真实 x_t）。每步都用真实前一帧刷新后验
（teacher forcing），因此是 10 个连续的"单步预测"，而非误差累积的多步 rollout。

产出：
  outputs/wm/vis/prediction_single_step.gif   10 帧动图，逐帧播放
  outputs/wm/vis/prediction_single_step.png   静态 2x10 对比网格（真实 / 预测）

用法（vla 环境）：
  python trainer/make_prediction_gif.py --ckpt outputs/wm/checkpoints/wm_step003000.pt
"""
import argparse
import glob
import math
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.utils import make_grid, save_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from eval_single_step import build_models


def parse_args():
    ap = argparse.ArgumentParser(description="生成单步预测对比 GIF")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/wm/vis"))
    ap.add_argument("--traj", default="traj_0002", help="用于可视化的轨迹 key（专家轨迹）")
    ap.add_argument("--start", type=int, default=2, help="从第几帧开始取 10 步")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_one_trajectory(hdf5_path, key, image_size, device):
    """加载单条轨迹，归一化口径与训练一致。"""
    with h5py.File(hdf5_path, "r") as f:
        low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
        high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
        span = torch.where(high == low, torch.ones_like(high), high - low)
        images = f[key]["images"][:]
        actions = f[key]["actions"][:]
    raw_t = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
    small = F.interpolate(raw_t, size=(image_size, image_size),
                          mode="bilinear", align_corners=False)
    imgs = (small / 127.5 - 1.0).half()                       # (n,3,S,S)
    a_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    acts = ((a_t - low) / span * 2 - 1).clamp(-1, 1)          # (n,4)
    return imgs, acts


def to_pil(tensor, size=192):
    """[-1,1] 张量 (3,H,W) -> RGB PIL 图像并放大。"""
    arr = tensor.detach().float().clamp(-1, 1).cpu().numpy()
    arr = ((arr + 1.0) * 127.5).astype(np.uint8)              # (3,H,W)
    img = Image.fromarray(np.transpose(arr, (1, 2, 0)), "RGB")
    return img.resize((size, size), Image.BILINEAR)


def psnr(pred, target):
    mse = F.mse_loss(pred.float(), target.float()).item()
    return 10.0 * math.log10(4.0 / (mse + 1e-8))


def main():
    args = parse_args()
    ckpt_path = args.ckpt
    if ckpt_path is None:
        cands = sorted(glob.glob(os.path.join(ROOT, "outputs/wm/checkpoints/wm_step*.pt")))
        ckpt_path = cands[-1]
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    print(f"[GIF] 检查点: {ckpt_path} step={ckpt.get('step')}", flush=True)
    M = build_models(args, ckpt)

    imgs, acts = load_one_trajectory(args.data, args.traj, 64, args.device)
    start, steps = args.start, args.steps
    assert start + steps <= imgs.shape[0], "取帧范围超出轨迹长度"
    os.makedirs(args.out_dir, exist_ok=True)

    # t=0：用真实首帧建立后验状态
    with torch.no_grad():
        e0 = M["enc"](imgs[start].float().unsqueeze(0))[0]
        state = M["rssm"].observe(
            M["rssm"].initial_state(1, device=args.device),
            acts[start].unsqueeze(0), e0)[0]

        real_frames, pred_frames = [], []
        labels = []
        for k in range(1, steps + 1):
            t = start + k
            # 单步预测：从真实前一帧的后验状态 + 动作 -> 先验 -> 解码
            ps, _ = M["rssm"].imagine_step(state, acts[t].unsqueeze(0))
            pred = M["dec"](M["rssm"].get_feature(ps))[0]
            real = imgs[t].float()
            real_frames.append(real)
            pred_frames.append(pred)
            labels.append((k, psnr(pred, real)))
            # 用真实 x_t 刷新后验，供下一步单步预测（teacher forcing）
            et = M["enc"](real.unsqueeze(0))[0]
            state, _ = M["rssm"].observe(state, acts[t].unsqueeze(0), et)

    # ---- 静态 2xN 对比网格（上=真实，下=单步预测）----
    real_stack = torch.stack(real_frames)   # (steps,3,64,64)
    pred_stack = torch.stack(pred_frames)
    grid = make_grid(torch.cat([real_stack, pred_stack], dim=0), nrow=steps,
                     normalize=True, value_range=(-1, 1), pad_value=0.4)
    png_path = os.path.join(args.out_dir, "prediction_single_step.png")
    save_image(grid, png_path)
    print(f"[GIF] 静态对比网格: {png_path}", flush=True)

    # ---- 动图：每帧 = [真实 | 单步预测] 并排 + 标题 ----
    panel, bar = 192, 34
    gif_frames = []
    for (real, pred), (k, p) in zip(zip(real_frames, pred_frames), labels):
        canvas = Image.new("RGB", (panel * 2 + 12, panel + bar), (32, 32, 38))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), f"t={k:02d}  REAL", fill=(235, 235, 235))
        draw.text((panel + 20, 10), f"PRED (1-step)  PSNR={p:.1f}dB", fill=(255, 214, 102))
        canvas.paste(to_pil(real, panel), (0, bar))
        canvas.paste(to_pil(pred, panel), (panel + 12, bar))
        gif_frames.append(canvas)

    gif_path = os.path.join(args.out_dir, "prediction_single_step.gif")
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=700, loop=0, disposal=2)
    print(f"[GIF] 动图: {gif_path} ({len(gif_frames)} 帧)", flush=True)
    ps = [p for _, p in labels]
    print(f"[GIF] 单步预测 PSNR: {np.mean(ps):.2f}dB (min {min(ps):.2f} / max {max(ps):.2f})", flush=True)


if __name__ == "__main__":
    main()
