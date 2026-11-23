#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""奖励头 / 价值头验证：给定轨迹，对比预测与真实的奖励曲线趋势。

验证内容：
1. 奖励预测曲线 vs 真实奖励曲线（归一化尺度 + 反归一化原始尺度），计算
   Pearson 相关系数、MAE；确认"趋势一致"（相关性显著为正）。
2. 价值预测曲线 vs 折扣回报目标曲线，计算相关系数与 MAE。
3. 生成对比图（两条曲线 + 散点），保存 PNG。

注意：验证用 observe 后验特征做单步预测（与训练口径一致），奖励/价值头
输入是看到真实帧后的隐状态，属于"诊断头是否学好"的口径，不是开环预测。

用法（vla 环境）：
  python trainer/verify_heads.py --ckpt outputs/wm_heads/checkpoints/wm_step004000.pt
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.vae import ConvEncoder, ConvDecoder
from models.rssm import RSSM
from models.heads import RewardHead, ValueHead


def parse_args():
    ap = argparse.ArgumentParser(description="奖励头/价值头验证")
    ap.add_argument("--ckpt", default=None, help="检查点；默认选 wm_heads 最新")
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/wm_heads"))
    ap.add_argument("--trajs", default="traj_0002,traj_0007",
                    help="用于画曲线的轨迹 key（逗号分隔，专家轨迹优先）")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--horizon", type=int, default=20, help="价值目标前瞻步数，与训练 seq_len 一致")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def build_models(args, ckpt):
    a = ckpt.get("args", {})
    latent = int(a.get("latent_dim", 64))
    stoch = int(a.get("stoch_dim", 64))
    det = int(a.get("det_dim", 200))
    feat = det + stoch
    enc = ConvEncoder(latent_dim=latent).to(args.device)
    rssm = RSSM(stoch_dim=stoch, det_dim=det, action_dim=4,
                embed_dim=latent).to(args.device)
    rh = RewardHead(feat).to(args.device)
    vh = ValueHead(feat).to(args.device)
    sd = ckpt["model"]
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    rssm.load_state_dict({k[len("rssm."):]: v for k, v in sd.items() if k.startswith("rssm.")})
    rh.load_state_dict({k[len("reward_head."):]: v for k, v in sd.items() if k.startswith("reward_head.")})
    vh.load_state_dict({k[len("value_head."):]: v for k, v in sd.items() if k.startswith("value_head.")})
    for m in (enc, rssm, rh, vh):
        m.eval()
    return dict(enc=enc, rssm=rssm, rh=rh, vh=vh, feat=feat)


def load_traj(f, key, image_size, device):
    low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
    high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
    span = torch.where(high == low, torch.ones_like(high), high - low)
    images = f[key]["images"][:]
    actions = f[key]["actions"][:]
    rewards = f[key]["rewards"][:]
    raw_t = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
    small = F.interpolate(raw_t, size=(image_size, image_size),
                          mode="bilinear", align_corners=False)
    imgs = small / 127.5 - 1.0
    a_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    acts = ((a_t - low) / span * 2 - 1).clamp(-1, 1)
    r_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    return imgs, acts, r_t


def discounted_returns_window(rewards, gamma, horizon=20):
    """每步起、截断到 horizon 步的折扣回报——与训练 20 步窗口口径一致。

    rewards: (T,) 归一化奖励
    返回:    (T,) G_t = sum_{k=t}^{min(T,t+horizon)-1} gamma^(k-t) * r_k
    """
    T = len(rewards)
    G = np.zeros(T, dtype=np.float64)
    for t in range(T):
        g, disc = 0.0, 1.0
        for k in range(t, min(T, t + horizon)):
            g += disc * rewards[k]
            disc *= gamma
        G[t] = g
    return G


def corr(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    args = parse_args()
    ckpt_path = args.ckpt
    if ckpt_path is None:
        cands = sorted(glob.glob(os.path.join(ROOT, "outputs/wm_heads/checkpoints/wm_step*.pt")))
        ckpt_path = cands[-1]
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    print(f"[HEADS] 检查点: {ckpt_path} step={ckpt.get('step')}", flush=True)
    M = build_models(args, ckpt)

    keys = [k.strip() for k in args.trajs.split(",") if k.strip()]
    os.makedirs(os.path.join(args.out_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)

    all_r_pred, all_r_true, all_v_pred, all_v_true = [], [], [], []
    per_traj = []

    with torch.no_grad(), h5py.File(args.data, "r") as f:
        avail = set(f.keys())
        keys = [k for k in keys if k in avail]
        if not keys:
            keys = sorted(avail)[:2]
        for key in keys:
            imgs, acts, r_raw = load_traj(f, key, 64, args.device)
            T = imgs.shape[0]
            r_norm = (r_raw / 10.0).clamp(-1, 1)               # 训练口径
            # 与训练口径一致：20 步窗口内折扣回报（避免整轨累计导致的尺度失配）
            v_target = discounted_returns_window(r_norm.cpu().numpy(), args.gamma,
                                                  args.horizon)  # (T,) numpy

            state = M["rssm"].initial_state(1, device=args.device)
            rp, vp = [], []
            for t in range(T):
                e = M["enc"](imgs[t].unsqueeze(0))[0]
                state, _ = M["rssm"].observe(state, acts[t].unsqueeze(0), e)
                feat = M["rssm"].get_feature(state)
                rp.append(M["rh"](feat).squeeze().item())
                vp.append(M["vh"](feat).squeeze().item())

            rp, vp = np.array(rp), np.array(vp)
            rt = r_norm.cpu().numpy()
            vt = np.asarray(v_target)
            c_r, c_v = corr(rp, rt), corr(vp, vt)
            mae_r = float(np.mean(np.abs(rp - rt)))
            mae_v = float(np.mean(np.abs(vp - vt)))
            per_traj.append({"traj": key, "reward_corr": round(c_r, 4),
                             "reward_mae": round(mae_r, 5),
                             "value_corr": round(c_v, 4), "value_mae": round(mae_v, 5),
                             "steps": T})
            all_r_pred.extend(rp.tolist()); all_r_true.extend(rt.tolist())
            all_v_pred.extend(vp.tolist()); all_v_true.extend(vt.tolist())
            print(f"[HEADS] {key}: reward corr={c_r:.4f} MAE={mae_r:.5f} | "
                  f"value corr={c_v:.4f} MAE={mae_v:.5f} ({T}步)", flush=True)

            # 单条轨迹曲线对比图
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=110)
            axes[0].plot(rt, label="True reward", color="#2ca02c", lw=1.6)
            axes[0].plot(rp, label="Predicted reward", color="#d62728", lw=1.6, alpha=0.85)
            axes[0].set_title(f"{key} Reward curve (normalized) corr={c_r:.3f}")
            axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlabel("step")
            axes[1].plot(vt, label="Discount-return target", color="#1f77b4", lw=1.6)
            axes[1].plot(vp, label="Value head prediction", color="#ff7f0e", lw=1.6, alpha=0.85)
            axes[1].set_title(f"Value curve corr={c_v:.3f}")
            axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xlabel("step")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "vis", f"head_curves_{key}.png"))
            plt.close(fig)

    summary = {
        "ckpt": ckpt_path, "gamma": args.gamma,
        "reward_corr": round(corr(all_r_pred, all_r_true), 4),
        "reward_mae": round(float(np.mean(np.abs(np.array(all_r_pred) - np.array(all_r_true)))), 5),
        "value_corr": round(corr(all_v_pred, all_v_true), 4),
        "value_mae": round(float(np.mean(np.abs(np.array(all_v_pred) - np.array(all_v_true)))), 5),
        "per_traj": per_traj,
    }
    with open(os.path.join(args.out_dir, "metrics", "heads_eval.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[HEADS] 汇总: reward corr={summary['reward_corr']} MAE={summary['reward_mae']} | "
          f"value corr={summary['value_corr']} MAE={summary['value_mae']}", flush=True)
    print("[HEADS] OK", flush=True)


if __name__ == "__main__":
    main()
