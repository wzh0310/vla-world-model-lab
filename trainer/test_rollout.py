# -*- coding: utf-8 -*-
"""Day 1 基础验证：WorldModel.rollout() 多步预测管线端到端测试。

读取 1 条专家轨迹：前 K 帧真实图像 + 对应动作做后验热身，再用后续 N 步动作
开环想象，输出 10 步预测帧序列，与真实帧逐 step 对齐计算 PSNR / SSIM，
并同步核对预测奖励与真实奖励的偏差。

产出（--out-dir，默认 outputs/rollout_test）：
  test_rollout.gif        真实 vs 预测并排动图
  test_rollout_grid.png   静态对比网格（上排真实 / 下排预测）
  test_rollout_metrics.json  逐 step PSNR/SSIM + 奖励误差汇总

用法（vla 环境）：
  python trainer/test_rollout.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt \
      --traj traj_0002 --warmup 3 --horizon 10
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
from skimage.metrics import structural_similarity as ssim_fn
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainer"))
from models.world_model import WorldModel
from make_prediction_gif import to_pil


def parse_args():
    ap = argparse.ArgumentParser(description="rollout() 多步预测管线验证")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/wm_tune1/checkpoints/wm_step004000.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/rollout_test"))
    ap.add_argument("--traj", default="traj_0002")
    ap.add_argument("--start", type=int, default=0, help="热身起始帧索引")
    ap.add_argument("--warmup", type=int, default=3, help="后验热身帧数 K")
    ap.add_argument("--horizon", type=int, default=10, help="开环预测步数 N")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_trajectory(data_path, traj):
    """读取轨迹并归一化：图像 224->64, [-1,1]；动作/奖励按 HDF5 attrs 归一化。"""
    with h5py.File(data_path, "r") as f:
        a_low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32))
        a_high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32))
        images = f[traj]["images"][:]      # (T,224,224,3) uint8
        actions = f[traj]["actions"][:]    # (T,4) 原始尺度
        rewards = f[traj]["rewards"][:]    # (T,) 原始尺度
    span = torch.where(a_high == a_low, torch.ones_like(a_high), a_high - a_low)
    acts = ((torch.as_tensor(actions, dtype=torch.float32) - a_low) / span * 2 - 1).clamp(-1, 1)
    # 奖励归一化与训练口径一致：clip(r/10, -1, 1)
    r_norm = (torch.as_tensor(rewards, dtype=torch.float32) / 10.0).clamp(-1, 1)
    return images, acts, r_norm


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)

    wm = WorldModel.from_checkpoint(args.ckpt, device=args.device)
    S = wm.image_size

    images_raw, acts, r_norm = load_trajectory(args.data, args.traj)
    raw = torch.as_tensor(images_raw).permute(0, 3, 1, 2).float()
    small = F.interpolate(raw, size=(S, S), mode="bilinear", align_corners=False)
    imgs = (small / 127.5 - 1.0)                                        # (T,3,S,S) [-1,1]
    T = imgs.shape[0]
    K, N = args.warmup, args.horizon
    if args.start + K + N > T:
        raise ValueError(f"start({args.start})+warmup({K})+horizon({N}) 超出轨迹长度 {T}")

    out = wm.rollout_from_trajectory(imgs, acts, warmup_steps=K, horizon=N,
                                     start=args.start, sample=True)
    preds = out["images"].cpu()                                         # (N,3,S,S) 移回 CPU 便于可视化
    reals = imgs[args.start + K: args.start + K + N]                    # (N,3,S,S)

    # 逐 step 对齐指标
    psnrs, ssims = [], []
    for k in range(N):
        mse = float(torch.mean((preds[k] - reals[k]) ** 2))
        psnrs.append(10.0 * math.log10(4.0 / (mse + 1e-8)))
        p = preds[k].cpu().numpy().transpose(1, 2, 0)
        r = reals[k].cpu().numpy().transpose(1, 2, 0)
        ssims.append(float(ssim_fn(r, p, channel_axis=2, data_range=2.0)))
    # 奖励对比（预测为归一化尺度，与真实归一化奖励按到达帧索引对齐）
    pred_r = out["rewards"].cpu() if out["rewards"] is not None else None
    real_r = r_norm[args.start + K: args.start + K + N]
    rew_mae = float(torch.mean(torch.abs(pred_r - real_r))) if pred_r is not None else None

    # 静态对比网格
    grid = make_grid(torch.cat([reals, preds], dim=0), nrow=N,
                     normalize=True, value_range=(-1, 1), pad_value=0.4)
    grid_path = os.path.join(args.out_dir, "test_rollout_grid.png")
    save_image(grid, grid_path)

    # 并排 GIF
    panel, bar = 176, 34
    frames = []
    for k in range(N):
        canvas = Image.new("RGB", (panel * 2 + 12, panel + bar), (32, 32, 38))
        d = ImageDraw.Draw(canvas)
        d.text((8, 9), f"REAL t={args.start + K + k:03d}  PSNR={psnrs[k]:.1f}dB", fill=(235, 235, 235))
        d.text((panel + 20, 9), f"PRED t={args.start + K + k:03d}", fill=(255, 214, 102))
        canvas.paste(to_pil(reals[k], panel), (0, bar))
        canvas.paste(to_pil(preds[k], panel), (panel + 12, bar))
        frames.append(canvas)
    gif_path = os.path.join(args.out_dir, "test_rollout.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=280, loop=0, disposal=2)

    # 汇总
    metrics = {
        "ckpt": args.ckpt, "traj": args.traj, "start": args.start,
        "warmup_steps": K, "horizon": N,
        "psnr_by_step": [round(x, 3) for x in psnrs],
        "ssim_by_step": [round(x, 4) for x in ssims],
        "psnr_mean": round(float(np.mean(psnrs)), 3),
        "psnr_min": round(float(np.min(psnrs)), 3),
        "ssim_mean": round(float(np.mean(ssims)), 4),
        "reward_mae_norm": round(rew_mae, 5) if rew_mae is not None else None,
        "reward_pred_norm_by_step": [round(float(x), 4) for x in pred_r] if pred_r is not None else None,
        "reward_real_norm_by_step": [round(float(x), 4) for x in real_r],
        "features_shape": list(out["features"].shape),
        "context_features_shape": list(out["context_features"].shape),
    }
    with open(os.path.join(args.out_dir, "test_rollout_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[TEST] 热身 {K} 帧（t={args.start}..{args.start + K - 1}）→ 开环预测 {N} 步"
          f"（t={args.start + K}..{args.start + K + N - 1}）", flush=True)
    for k in range(N):
        print(f"  step{k + 1:02d}: PSNR={psnrs[k]:6.2f}dB  SSIM={ssims[k]:.3f}", flush=True)
    print(f"[TEST] PSNR mean={metrics['psnr_mean']} min={metrics['psnr_min']} | "
          f"SSIM mean={metrics['ssim_mean']} | 奖励MAE(norm)={metrics['reward_mae_norm']}", flush=True)
    ok = metrics["psnr_min"] >= 18.0 and metrics["psnr_mean"] >= 21.0
    print(f"[TEST] 崩坏检查: {'PASS（全程无崩坏）' if ok else 'WARN（存在低质量步，请人工查看 GIF）'}", flush=True)
    print(f"[TEST] GIF: {gif_path}", flush=True)
    print(f"[TEST] 网格: {grid_path}", flush=True)


if __name__ == "__main__":
    main()
