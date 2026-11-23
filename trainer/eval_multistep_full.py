#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多步长全面评估：1/5/10/20 步 PSNR 衰减 + 各步长奖励预测误差 + 可视化素材。

统一口径：多条专家轨迹、固定起点集合，从真实首帧建后验起点，随后 N 步
开环想象（imagine_rollout，全程不接真实图像），逐 step 对齐真实帧 / 真实奖励：

  - 图像：pred_img[k] vs real_img[st+1+k] -> PSNR(dB) / SSIM
  - 奖励：head(feature(state[k])) vs real_rew[st+1+k]（归一化 r/10 尺度）
          -> MAE / RMSE，同时给出原始奖励尺度（×10）

产出（均写入 --out-dir）：
  metrics/multistep_full_eval.json  逐 step + 关键步(1/5/10/20) 汇总
  vis/psnr_decay.png                PSNR/SSIM 随步长衰减折线图（英文标签）
  vis/traj_<key>_20step.gif         3 组「真实 vs 20 步预测」并排对比视频
  vis/traj_<key>_compare.png        对应静态 2xN 网格（便于查看）

用法（vla 环境）：
  python trainer/eval_multistep_full.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainer"))
from eval_single_step import build_models
from make_prediction_gif import to_pil


def parse_args():
    ap = argparse.ArgumentParser(description="多步长全面评估（PSNR 衰减 + 奖励误差 + 可视化）")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/wm_tune1/checkpoints/wm_step004000.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/wm_tune1"))
    ap.add_argument("--trajs", default="traj_0002,traj_0007,traj_0011,traj_0015,traj_0020")
    ap.add_argument("--starts", default="2,8")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--key-steps", default="1,5,10,20")
    ap.add_argument("--vis-trajs", default="traj_0002,traj_0007,traj_0011")
    ap.add_argument("--vis-start", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_trajectory_full(hdf5_path, key, image_size, device):
    """加载单条轨迹：图像/动作/奖励，归一化口径与 fast_loader 完全一致。

    返回：
      imgs (n,3,S,S) float16 [-1,1]
      acts (n,4)     float32 [-1,1]
      rews (n,)      float32  归一化奖励 r/10 clamp[-1,1]
    """
    with h5py.File(hdf5_path, "r") as f:
        low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
        high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
        span = torch.where(high == low, torch.ones_like(high), high - low)
        images = f[key]["images"][:]
        actions = f[key]["actions"][:]
        rewards = f[key]["rewards"][:]
    raw = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
    small = F.interpolate(raw, size=(image_size, image_size), mode="bilinear", align_corners=False)
    imgs = (small / 127.5 - 1.0).half()
    a_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    acts = ((a_t - low) / span * 2 - 1).clamp(-1, 1)
    r_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    rews = (r_t / 10.0).clamp(-1, 1)
    return imgs, acts, rews


def psnr_of(pred, target):
    mse = F.mse_loss(pred.float(), target.float()).item()
    return 10.0 * math.log10(4.0 / (mse + 1e-8))


def rollout_from(M, imgs, acts, st, steps, device):
    """从 imgs[st] 建后验起点，做 steps 步开环想象，返回预测帧与预测奖励。"""
    enc, dec, rssm, head = M["enc"], M["dec"], M["rssm"], M["head"]
    e0 = enc(imgs[st].float().unsqueeze(0))[0]
    init = rssm.observe(rssm.initial_state(1, device=device), acts[st].unsqueeze(0), e0)[0]
    act_seq = acts[st + 1:st + 1 + steps].unsqueeze(0)     # (1,steps,4)
    states, _ = rssm.imagine_rollout(init, act_seq)        # 全开环，不接真实图像
    preds = rssm.decode_rollout(states, dec)[:, 0]        # (steps,3,S,S)
    feats = torch.stack([rssm.get_feature(s) for s in states])[:, 0]  # (steps,feat)
    pred_r = head(feats).squeeze(-1)                      # (steps,) 归一化奖励
    return preds, pred_r


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    M = build_models(args, ckpt)
    steps = args.steps
    key_steps = [int(x) for x in args.key_steps.split(",") if x.strip()]
    key_steps = [k for k in key_steps if 1 <= k <= steps]

    trajs = [t.strip() for t in args.trajs.split(",") if t.strip()]
    starts = [int(s) for s in args.starts.split(",") if s.strip()]
    per_step = {"psnr": [[] for _ in range(steps)], "ssim": [[] for _ in range(steps)],
                "rew_ae": [[] for _ in range(steps)]}   # rew_ae: 预测-真实（归一化尺度）

    with torch.no_grad(), h5py.File(args.data, "r") as f:
        avail = set(f.keys())
        trajs = [t for t in trajs if t in avail]
        for key in trajs:
            imgs, acts, rews = load_trajectory_full(args.data, key, 64, args.device)
            n = imgs.shape[0]
            for st in starts:
                if st + steps >= n:     # 需要 real[st+1 .. st+steps]
                    continue
                preds, pred_r = rollout_from(M, imgs, acts, st, steps, args.device)
                for k in range(steps):
                    real_img = imgs[st + 1 + k].float()
                    per_step["psnr"][k].append(psnr_of(preds[k], real_img))
                    a = preds[k].float().clamp(-1, 1).cpu().numpy().transpose(1, 2, 0)
                    b = real_img.clamp(-1, 1).cpu().numpy().transpose(1, 2, 0)
                    per_step["ssim"][k].append(float(ssim(b, a, channel_axis=2, data_range=2.0)))
                    real_r = rews[st + 1 + k]
                    per_step["rew_ae"][k].append((pred_r[k].detach().float() - real_r).item())
        n_eval = len(per_step["psnr"][0]) if per_step["psnr"][0] else 0

    if n_eval == 0:
        raise RuntimeError("没有任何有效开环轨迹，请检查 --starts 与轨迹长度")

    psnr_mean = [float(np.mean(x)) for x in per_step["psnr"]]
    psnr_std = [float(np.std(x)) for x in per_step["psnr"]]
    ssim_mean = [float(np.mean(x)) for x in per_step["ssim"]]
    rew_mae = [float(np.mean(np.abs(x))) for x in per_step["rew_ae"]]
    rew_rmse = [float(np.sqrt(np.mean(np.square(x)))) for x in per_step["rew_ae"]]

    def at_step(arr, k):
        return round(float(arr[k - 1]), 4)

    key_summary = {}
    for k in key_steps:
        key_summary[f"step{k}"] = {
            "psnr_db": at_step(psnr_mean, k),
            "psnr_std": at_step(psnr_std, k),
            "ssim": at_step(ssim_mean, k),
            "reward_mae_norm": at_step(rew_mae, k),
            "reward_mae_raw10x": round(at_step(rew_mae, k) * 10.0, 4),
            "reward_rmse_norm": at_step(rew_rmse, k),
            "reward_rmse_raw10x": round(at_step(rew_rmse, k) * 10.0, 4),
        }
    summary = {
        "ckpt": args.ckpt, "stoch": M["stoch"], "det": M["det"],
        "n_eval_rollouts": n_eval, "steps": steps, "trajs": trajs, "starts": starts,
        "psnr_mean_by_step": [round(x, 3) for x in psnr_mean],
        "psnr_std_by_step": [round(x, 3) for x in psnr_std],
        "ssim_mean_by_step": [round(x, 4) for x in ssim_mean],
        "reward_mae_norm_by_step": [round(x, 4) for x in rew_mae],
        "reward_rmse_norm_by_step": [round(x, 4) for x in rew_rmse],
        "key_steps": key_summary,
        "psnr_step1": round(psnr_mean[0], 3),
        "psnr_step20": round(psnr_mean[-1], 3),
        "psnr_drop_1to20": round(psnr_mean[0] - psnr_mean[-1], 3),
    }
    vis_dir = os.path.join(args.out_dir, "vis")
    met_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)
    with open(os.path.join(met_dir, "multistep_full_eval.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[FULL] 评估 {n_eval} 条开环 × {steps} 步", flush=True)
    for k in key_steps:
        s = key_summary[f"step{k}"]
        print(f"[FULL] step{k:>2}: PSNR={s['psnr_db']:.2f}dB SSIM={s['ssim']:.4f} "
              f"rewMAE(norm)={s['reward_mae_norm']:.4f}", flush=True)
    print(f"[FULL] PSNR 1->20 衰减 {summary['psnr_drop_1to20']:.2f}dB", flush=True)

    # ---------- 折线图：PSNR / SSIM 随步长衰减（英文标签，容器无中文字体）----------
    xs = np.arange(1, steps + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(xs, psnr_mean, color="#d62728", lw=2.0, marker="o", ms=3, label="PSNR (dB)")
    ax1.fill_between(xs, np.array(psnr_mean) - np.array(psnr_std),
                     np.array(psnr_mean) + np.array(psnr_std), color="#d62728", alpha=0.15)
    ax1.set_xlabel("Prediction step (open-loop)")
    ax1.set_ylabel("PSNR (dB)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(alpha=0.3)
    for k in key_steps:
        if k <= steps:
            ax1.axvline(k, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot(xs, ssim_mean, color="#1f77b4", lw=1.8, ls="--", label="SSIM")
    ax2.set_ylabel("SSIM", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0, 1.0)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Open-loop prediction quality vs horizon")
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "psnr_decay.png"), dpi=130)
    plt.close(fig)
    print(f"[FULL] 折线图: {vis_dir}/psnr_decay.png", flush=True)

    # ---------- 3 组轨迹：真实 vs 20 步预测并排 GIF + 静态网格 ----------
    from torchvision.utils import make_grid, save_image
    for key in [t.strip() for t in args.vis_trajs.split(",") if t.strip()]:
        if key not in avail:
            continue
        imgs, acts, rews = load_trajectory_full(args.data, key, 64, args.device)
        st = args.vis_start
        if st + steps >= imgs.shape[0]:
            print(f"[FULL] skip vis {key}: 轨迹过短", flush=True)
            continue
        with torch.no_grad():
            preds, _ = rollout_from(M, imgs, acts, st, steps, args.device)
        reals = imgs[st + 1:st + 1 + steps].float()

        grid = make_grid(torch.cat([reals, preds], dim=0), nrow=steps,
                         normalize=True, value_range=(-1, 1), pad_value=0.4)
        save_image(grid, os.path.join(vis_dir, f"{key}_compare.png"))

        panel, bar = 176, 30
        frames = []
        for k in range(steps):
            canvas = Image.new("RGB", (panel * 2 + 12, panel + bar), (32, 32, 38))
            d = ImageDraw.Draw(canvas)
            d.text((8, 9), f"{key} REAL  t={k + 1:02d}", fill=(235, 235, 235))
            d.text((panel + 20, 9), f"PRED t={k + 1:02d}  PSNR={psnr_mean[k]:.1f}dB",
                   fill=(255, 214, 102))
            canvas.paste(to_pil(reals[k], panel), (0, bar))
            canvas.paste(to_pil(preds[k], panel), (panel + 12, bar))
            frames.append(canvas)
        frames[0].save(os.path.join(vis_dir, f"{key}_20step.gif"),
                       save_all=True, append_images=frames[1:], duration=260,
                       loop=0, disposal=2)
        print(f"[FULL] 可视化: {key}_20step.gif / {key}_compare.png", flush=True)
    print("[FULL] OK", flush=True)


if __name__ == "__main__":
    main()
