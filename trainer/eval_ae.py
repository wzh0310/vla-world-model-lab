#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自编码器效果验收：批量重建 + PSNR 定量指标 + 对比图。

用法（vla 环境）：
  python trainer/eval_ae.py --ckpt outputs/checkpoints/ae_step004000.pt --num-samples 200
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, ROOT)
from episodic_dataset import EpisodicDataset
from models.vae import ConvVAE


def psnr(mse: float) -> float:
    """图像归一化到 [-1,1]，PSNR = 10*log10(4/MSE)（峰值 2^2=4）。"""
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(4.0 / mse)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="检查点路径；不传则自动选 step 最大的")
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/ae_eval"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = args.ckpt
    if ckpt is None:
        candidates = sorted(glob.glob(os.path.join(ROOT, "outputs/checkpoints/ae_step*.pt")))
        best = os.path.join(ROOT, "outputs/checkpoints/ae_best.pt")
        if os.path.exists(best):
            candidates.append(best)
        if not candidates:
            raise FileNotFoundError("没有找到检查点")
        ckpt = candidates[-1]
    state = torch.load(ckpt, map_location=args.device, weights_only=False)
    cfg = state.get("args", {})
    latent_dim = cfg.get("latent_dim", 32)
    model = ConvVAE(latent_dim=latent_dim).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[EVAL] 检查点: {ckpt}  latent={latent_dim}  step={state.get('step')}")

    ds = EpisodicDataset(args.data, seq_len=20, image_size=64)
    n = min(args.num_samples, len(ds))
    mses, psnrs = [], []
    vis_frames = []
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            x = torch.from_numpy(item["image_seq"]).to(args.device)  # (20,3,64,64)
            recon = model(x)["recon"]
            mse = torch.nn.functional.mse_loss(recon, x).item()
            mses.append(mse)
            psnrs.append(psnr(mse))
            if i < 2:  # 取两组序列拼对比图
                vis_frames.append((x[:8].cpu(), recon[:8].cpu()))

    mses, psnrs = np.array(mses), np.array(psnrs)
    print(f"[EVAL] 样本数={n}  MSE 均值={mses.mean():.6f}  中位数={np.median(mses):.6f}")
    print(f"[EVAL] PSNR 均值={psnrs.mean():.2f} dB  中位数={np.median(psnrs):.2f} dB  "
          f"最小={psnrs.min():.2f} dB")

    os.makedirs(args.out_dir, exist_ok=True)
    for gi, (orig, rec) in enumerate(vis_frames):
        grid = make_grid(torch.cat([orig, rec], dim=0), nrow=8,
                         normalize=True, value_range=(-1, 1), pad_value=0.3)
        path = os.path.join(args.out_dir, f"recon_compare_{gi}.png")
        save_image(grid, path)
        print(f"[EVAL] 对比图: {path}")

    metrics = {"checkpoint": ckpt, "step": state.get("step"), "latent_dim": latent_dim,
               "num_samples": n, "mse_mean": float(mses.mean()),
               "mse_median": float(np.median(mses)), "psnr_mean": float(psnrs.mean()),
               "psnr_median": float(np.median(psnrs)), "psnr_min": float(psnrs.min())}
    import json
    json_path = os.path.join(args.out_dir, "ae_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[EVAL] 指标已保存: {json_path}")


if __name__ == "__main__":
    main()
