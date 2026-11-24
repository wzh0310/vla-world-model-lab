#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多步预测对比演示：3 条测试轨迹的「真实帧 vs 预测帧」并排视频。

用世界模型统一推理封装（WorldModel.rollout_from_trajectory）：
  - 前 warmup 帧真实图像建后验起点，随后 20 步纯开环想象（不接真实帧）；
  - 每步生成「真实 | 预测」并排帧，逐帧标注 PSNR/SSIM；
  - 在 5 / 10 / 20 步关键节点用高亮横幅强调，直观体现精度随步长衰减；
  - 每条轨迹额外输出 5/10/20 三步的三联静态对比图。

产出（写入 --out-dir，默认 results/visualization/）：
  compare_<traj>.gif       20 步并排对比视频（标注关键节点）
  keypoints_<traj>.png     5/10/20 步静态三联图（真实/预测/误差热图）
  decay_<traj>.png         该轨迹 PSNR/SSIM 随步长衰减曲线

用法（vla 环境）：
  python scripts/compare_video.py --ckpt outputs/wm_mixtrain/checkpoints/wm_best.pt \
      --data outputs/data/wm_dataset/test.h5 --out-dir results/visualization
"""
import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.world_model import WorldModel

KEY_STEPS = (5, 10, 20)


def parse_args():
    ap = argparse.ArgumentParser(description="多步预测对比演示视频")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/wm_mixtrain/checkpoints/wm_best.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/wm_dataset/test.h5"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/visualization"))
    ap.add_argument("--trajs", default="traj_0002,traj_0007,traj_0011")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def load_traj(hdf5_path, key, image_size, device):
    with h5py.File(hdf5_path, "r") as f:
        low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
        high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
        span = torch.where(high == low, torch.ones_like(high), high - low)
        images = f[key]["images"][:]
        actions = f[key]["actions"][:]
    raw = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
    small = F.interpolate(raw, size=(image_size, image_size), mode="bilinear", align_corners=False)
    imgs = (small / 127.5 - 1.0)                      # [-1,1]
    a = torch.as_tensor(actions, dtype=torch.float32, device=device)
    acts = ((a - low) / span * 2 - 1).clamp(-1, 1)
    return imgs, acts


def to_pil(img, size):
    """(3,H,W) [-1,1] -> PIL RGB。"""
    arr = img.detach().float().clamp(-1, 1).cpu().permute(1, 2, 0).numpy()
    arr = ((arr + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.BILINEAR)


def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 10.0 * math.log10(4.0 / (mse + 1e-8))


def ssim_of(a, b):
    x = a.detach().float().clamp(-1, 1).cpu().permute(1, 2, 0).numpy()
    y = b.detach().float().clamp(-1, 1).cpu().permute(1, 2, 0).numpy()
    return float(ssim(y, x, channel_axis=2, data_range=2.0))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    wm = WorldModel.from_checkpoint(args.ckpt, device=args.device)
    S = wm.image_size
    panel = 176
    bar = 26
    summary = {}
    # 关键节点随 --steps 自适应：短步长预测时退化为最后一步，避免索引越界。
    key_steps = [k for k in KEY_STEPS if k <= args.steps] or [args.steps]

    for key in [t.strip() for t in args.trajs.split(",") if t.strip()]:
        imgs, acts = load_traj(args.data, key, S, args.device)
        need = args.warmup + args.steps
        if imgs.shape[0] < need:
            print(f"[VIZ] 跳过 {key}: 帧数不足 {imgs.shape[0]}<{need}", flush=True)
            continue
        with torch.no_grad():
            out = wm.rollout_from_trajectory(imgs, acts, warmup_steps=args.warmup,
                                             horizon=args.steps, sample=False)
        preds = out["images"]                        # (steps,3,S,S)
        reals = imgs[args.warmup:args.warmup + args.steps]
        p_list, s_list = [], []
        for k in range(args.steps):
            p_list.append(psnr(preds[k], reals[k]))
            s_list.append(ssim_of(preds[k], reals[k]))

        # ---- 并排 GIF：真实 | 预测，逐帧指标，关键节点高亮 ----
        frames = []
        for k in range(args.steps):
            step_no = k + 1
            is_key = step_no in key_steps
            bg = (60, 44, 12) if is_key else (30, 30, 36)
            canvas = Image.new("RGB", (panel * 2 + 14, panel + bar + (16 if is_key else 0)), bg)
            d = ImageDraw.Draw(canvas)
            tag = "  << KEY" if is_key else ""
            d.text((6, 6), f"{key}  REAL  t={step_no:02d}", fill=(235, 235, 235))
            d.text((panel + 20, 6),
                   f"PRED t={step_no:02d}  PSNR={p_list[k]:.1f} SSIM={s_list[k]:.2f}{tag}",
                   fill=(255, 214, 102) if is_key else (200, 200, 210))
            canvas.paste(to_pil(reals[k], panel), (0, bar))
            canvas.paste(to_pil(preds[k], panel), (panel + 14, bar))
            frames.append(canvas)
        gif_path = os.path.join(args.out_dir, f"compare_{key}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=280, loop=0, disposal=2)

        # ---- 关键节点三联图：真实 | 预测 | 绝对误差热图 ----
        kp = [k - 1 for k in key_steps]
        fig, axes = plt.subplots(len(kp), 3, figsize=(10, 3.1 * len(kp)))
        if len(kp) == 1:
            axes = axes[None, :]
        for r, k in enumerate(kp):
            err = (preds[k].float() - reals[k].float()).abs().mean(0).cpu().numpy()
            row = [reals[k], preds[k]]
            for c in range(2):
                arr = row[c].detach().float().clamp(-1, 1).cpu().permute(1, 2, 0).numpy()
                axes[r, c].imshow((arr + 1) / 2)
                axes[r, c].set_title(("REAL" if c == 0 else "PRED") +
                                     f" @ step {k+1}\nPSNR={p_list[k]:.1f}  SSIM={s_list[k]:.3f}")
                axes[r, c].axis("off")
            im = axes[r, 2].imshow(err, cmap="magma", vmin=0, vmax=1)
            axes[r, 2].set_title(f"|ERR| @ step {k+1}\nmean={err.mean():.3f}")
            axes[r, 2].axis("off")
            plt.colorbar(im, ax=axes[r, 2], fraction=0.046)
        fig.tight_layout()
        kp_path = os.path.join(args.out_dir, f"keypoints_{key}.png")
        fig.savefig(kp_path, dpi=130)
        plt.close(fig)

        # ---- 衰减曲线 ----
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        xs = np.arange(1, args.steps + 1)
        ax1.plot(xs, p_list, color="#d62728", lw=2, marker="o", ms=3, label="PSNR (dB)")
        ax1.set_xlabel("Open-loop step"); ax1.set_ylabel("PSNR (dB)", color="#d62728")
        ax1.tick_params(axis="y", labelcolor="#d62728"); ax1.grid(alpha=0.3)
        for k in key_steps:
            ax1.axvline(k, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax2 = ax1.twinx()
        ax2.plot(xs, s_list, color="#1f77b4", lw=1.8, ls="--", label="SSIM")
        ax2.set_ylabel("SSIM", color="#1f77b4"); ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.set_ylim(0, 1)
        ax1.set_title(f"{key}: prediction quality vs horizon")
        fig.tight_layout()
        dc_path = os.path.join(args.out_dir, f"decay_{key}.png")
        fig.savefig(dc_path, dpi=130); plt.close(fig)

        summary[key] = {
            "psnr_by_step": [round(x, 3) for x in p_list],
            "ssim_by_step": [round(x, 4) for x in s_list],
            "keypoints": {f"step{k}": {"psnr": round(p_list[k - 1], 3),
                                       "ssim": round(s_list[k - 1], 4)} for k in key_steps},
        }
        _brief = " ".join(f"{k}步={p_list[k - 1]:.1f}" for k in key_steps)
        print(f"[VIZ] {key} 完成: PSNR 1步={p_list[0]:.1f} {_brief} dB", flush=True)

    with open(os.path.join(args.out_dir, "compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[VIZ] 完成，产出目录: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
