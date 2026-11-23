#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""聚焦立方体的开环预测可视化：放大中央区域，检验立方体位置是否可辨。

对未来作对比：上排真实帧（中央裁剪放大）、下排开环预测帧（中央裁剪放大）。
立方体（黄绿色小方块）位于画面中央偏左，64x64 下仅约 3x3 像素，故裁剪
中央 48x48 区域放大到 288x288 便于观察。同时输出完整帧的对照。
"""
import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image, make_grid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainer"))
from eval_single_step import build_models
from make_prediction_gif import load_one_trajectory, to_pil


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj", default="traj_0002")
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    a = parse_args()
    ckpt = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    M = build_models(a, ckpt)
    enc, dec, rssm = M["enc"], M["dec"], M["rssm"]
    imgs, acts = load_one_trajectory(a.data, a.traj, 64, a.device)
    st = a.start
    with torch.no_grad():
        e0 = enc(imgs[st].float().unsqueeze(0))[0]
        init = rssm.observe(rssm.initial_state(1, device=a.device),
                            acts[st].unsqueeze(0), e0)[0]
        states, _ = rssm.imagine_rollout(init, acts[st + 1:st + 1 + a.steps].unsqueeze(0))
        preds = rssm.decode_rollout(states, dec)[:, 0]
        reals = imgs[st + 1:st + 1 + a.steps].float()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # 完整帧对照图
    grid = make_grid(torch.cat([reals, preds], dim=0), nrow=a.steps,
                     normalize=True, value_range=(-1, 1), pad_value=0.4)
    save_image(grid, a.out.replace(".png", "_full.png"))

    # 中央区域放大对照：每格 = [真实放大 | 预测放大]，逐帧
    panel = 288
    bar = 30
    frames = []
    for k in range(a.steps):
        real_pil = to_pil(reals[k], 192)   # 192x192 from 64
        pred_pil = to_pil(preds[k], 192)
        # 中央裁剪：192 图的中央 144x144 -> 放大到 panel
        cx0, cy0 = (192 - 144) // 2, (192 - 144) // 2
        real_c = real_pil.crop((cx0, cy0, cx0 + 144, cy0 + 144)).resize((panel, panel), Image.BILINEAR)
        pred_c = pred_pil.crop((cx0, cy0, cx0 + 144, cy0 + 144)).resize((panel, panel), Image.BILINEAR)
        canvas = Image.new("RGB", (panel * 2 + 10, panel + bar), (32, 32, 38))
        d = ImageDraw.Draw(canvas)
        d.text((8, 8), f"REAL step {k + 1}", fill=(235, 235, 235))
        d.text((panel + 18, 8), f"OPEN-LOOP step {k + 1}", fill=(255, 214, 102))
        canvas.paste(real_c, (0, bar))
        canvas.paste(pred_c, (panel + 10, bar))
        frames.append(canvas)
    frames[0].save(a.out, save_all=True, append_images=frames[1:],
                   duration=800, loop=0, disposal=2)
    frames[0].save(a.out.replace(".gif", "_step1.png"))
    frames[-1].save(a.out.replace(".gif", "_step10.png"))
    print(f"[FOCUS] saved {a.out} (+ full/step1/step10 variants)", flush=True)


if __name__ == "__main__":
    main()
