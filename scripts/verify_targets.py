#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""世界模型达标验证：在 9:1 测试集上用最优权重跑开环 20 步，核验两项达标目标。

达标目标：
  1) 20 步预测 SSIM ≥ 0.6
  2) 奖励预测误差 < 0.5（原始奖励尺度；同时报告归一化尺度口径）

运行容器环境：vla。
用法：
  python scripts/verify_targets.py --ckpt outputs/wm_mixtrain/checkpoints/wm_best.pt \
      --data outputs/data/wm_dataset/test.h5
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.world_model import WorldModel  # noqa: E402
from trainer.metrics import psnr_metric, reward_mae, ssim_metric  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="世界模型达标验证")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/wm_mixtrain/checkpoints/wm_best.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/wm_dataset/test.h5"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n-traj", type=int, default=10, help="测试轨迹数（不足则取全部）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-json", default=os.path.join(ROOT, "outputs/wm_mixtrain/verify_report.json"))
    ap.add_argument("--reward-threshold", type=float, default=0.5)
    ap.add_argument("--ssim-threshold", type=float, default=0.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_episodes(data, n_traj, steps, seed):
    rng = np.random.default_rng(seed)
    with h5py.File(data, "r") as f:
        keys = sorted(f.keys())
        pick = list(rng.permutation(len(keys))[:n_traj])
        a_low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32))
        a_high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32))
        span = torch.where(a_high == a_low, torch.ones_like(a_high), a_high - a_low)
        eps = []
        for i in pick:
            g = f[keys[i]]
            imgs = g["images"][:]
            # 选一个起点，保证有 steps+1 帧
            if imgs.shape[0] < steps + 1:
                continue
            start = int(rng.integers(0, imgs.shape[0] - steps))
            raw = torch.as_tensor(imgs[start:start + steps + 1]).permute(0, 3, 1, 2).float()
            small = F.interpolate(raw, size=(64, 64), mode="bilinear", align_corners=False)
            imgs_t = small / 127.5 - 1.0
            acts = ((torch.as_tensor(g["actions"][:], dtype=torch.float32) - a_low)
                    / span * 2 - 1).clamp(-1, 1)[start:start + steps]
            rws = torch.as_tensor(g["rewards"][:], dtype=torch.float32)[start:start + steps]
            eps.append({"images": imgs_t, "actions": acts, "rewards": rws})
    return eps


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wm = WorldModel.from_checkpoint(args.ckpt, device=args.device)
    eps = load_episodes(args.data, args.n_traj, args.steps, args.seed)
    if not eps:
        raise SystemExit("无可用测试轨迹（长度不足）")

    ssims, psnrs, rew_norm, rew_raw = [], [], [], []
    for ep in eps:
        imgs, acts, rws = ep["images"], ep["actions"], ep["rewards"]
        dev = wm.device
        st = wm.rssm.initial_state(1, device=dev)
        st, _ = wm.rssm.observe(st, acts[0].to(dev), wm._encode(imgs[0].to(dev)))
        states, _ = wm.rssm.imagine_rollout(st, acts.to(dev), sample=False)
        preds = wm.rssm.decode_rollout(states, wm.decoder).clamp(-1, 1)[:, 0].cpu()
        reals = imgs[1:args.steps + 1]
        ssims.append(ssim_metric(preds, reals).mean())
        psnrs.append(psnr_metric(preds, reals).mean())
        feat = torch.stack([wm.rssm.get_feature(s).reshape(-1) for s in states])
        pred_r = wm.reward_head(feat.to(dev)).squeeze(-1).cpu()
        rew_norm.append((pred_r - (rws / 10.0).clamp(-1, 1)).abs().mean())
        rew_raw.append((pred_r * 10.0 - rws).abs().mean())

    def m(x):
        return float(np.mean(np.asarray(x, dtype=np.float64)))

    ssim_m, psnr_m, rew_norm_m, rew_raw_m = m(ssims), m(psnrs), m(rew_norm), m(rew_raw)
    ssim_ok = ssim_m >= args.ssim_threshold
    rew_ok = rew_raw_m < args.reward_threshold
    report = {
        "ckpt": args.ckpt, "data": args.data, "n_traj": len(eps), "steps": args.steps,
        "ssim20": round(ssim_m, 4), "psnr20": round(psnr_m, 3),
        "reward_mae_normalized": round(rew_norm_m, 5),
        "reward_mae_raw": round(rew_raw_m, 4),
        "targets": {"ssim20_min": args.ssim_threshold, "reward_mae_raw_max": args.reward_threshold},
        "pass_ssim": ssim_ok, "pass_reward": rew_ok,
        "overall_pass": bool(ssim_ok and rew_ok),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"[达标验证] {len(eps)} 条测试轨迹 × 开环 {args.steps} 步")
    print(f"  SSIM@{args.steps}      = {ssim_m:.4f}  (目标 ≥ {args.ssim_threshold})  "
          f"{'PASS' if ssim_ok else 'FAIL'}")
    print(f"  PSNR@{args.steps}      = {psnr_m:.3f} dB")
    print(f"  奖励MAE(归一化) = {rew_norm_m:.5f}")
    print(f"  奖励MAE(原始)   = {rew_raw_m:.4f}  (目标 < {args.reward_threshold})  "
          f"{'PASS' if rew_ok else 'FAIL'}")
    print(f"  总体: {'PASS 全部达标' if ssim_ok and rew_ok else 'FAIL'}")
    print(f"  报告: {args.out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
