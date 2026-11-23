#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""开环多步预测基础测试。

核心验证：RSSM.imagine_rollout(initial_state, actions) 从真实首帧编码得到的
初始状态出发，只接收动作序列，全程不接入真实图像，自回归生成 N 步隐状态；
再用 decode_rollout 批量解码为图像序列，可生成任意步长预测轨迹。

三种口径对照：
  - 真实序列：数据集真实帧，只用于评估/可视化，不输入后续步骤。
  - 开环预测（测试对象）：仅 t=0 用真实首帧建后验，之后 N 步纯动力学想象。
  - 单步预测（teacher forcing，参考上界）：每步用真实前一帧刷新后验再预测一步。

产出（位于 --out-dir）：
  vis/open_loop_rollout.gif  逐帧 [真实 | 开环预测] 动图
  vis/open_loop_rollout.png  3xN 静态对比（真实 / 开环 / 单步）
  metrics/open_loop_eval.json 逐步 PSNR/SSIM、维度检查、误差累积摘要

用法（vla 环境）：
  python trainer/test_open_loop.py --ckpt outputs/wm_fb02/checkpoints/wm_step004000.pt
"""
import argparse
import glob
import json
import math
import os
import sys

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
    ap = argparse.ArgumentParser(description="开环多步预测基础测试")
    ap.add_argument("--ckpt", default=None, help="世界模型检查点；默认选 wm_fb02 最新")
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/wm_fb02"))
    ap.add_argument("--traj", default="traj_0002", help="可视化用轨迹 key（专家轨迹）")
    ap.add_argument("--start", type=int, default=2, help="从第几帧建立初始状态")
    ap.add_argument("--steps", type=int, default=10, help="开环预测步数（可任意指定）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        cands = sorted(glob.glob(os.path.join(ROOT, "outputs/wm_fb02/checkpoints/wm_step*.pt")))
        ckpt_path = cands[-1]
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    print(f"[OPEN] 检查点: {ckpt_path} step={ckpt.get('step')}", flush=True)
    M = build_models(args, ckpt)
    enc, dec, rssm = M["enc"], M["dec"], M["rssm"]

    imgs, acts = load_one_trajectory(args.data, args.traj, 64, args.device)
    start, steps = args.start, args.steps
    assert start + steps < imgs.shape[0], "取帧范围超出轨迹长度"

    with torch.no_grad():
        # t=0：唯一一次接入真实图像——编码首帧得到后验起点
        e0 = enc(imgs[start].float().unsqueeze(0))[0]
        init = rssm.observe(rssm.initial_state(1, device=args.device),
                            acts[start].unsqueeze(0), e0)[0]

        # (A) 开环 rollout：之后完全不给真实帧，只喂动作序列
        action_seq = acts[start + 1:start + 1 + steps].unsqueeze(0)   # (1,N,4)
        states, infos = rssm.imagine_rollout(init, action_seq)
        open_preds = rssm.decode_rollout(states, dec)[:, 0]           # (N,3,64,64)

        # (B) teacher-forcing 单步预测（参考上界）
        tf_preds, s_tf = [], init
        for k in range(steps):
            t = start + 1 + k
            ps, _ = rssm.imagine_step(s_tf, acts[t].unsqueeze(0))
            tf_preds.append(dec(rssm.get_feature(ps))[0])
            et = enc(imgs[t].float().unsqueeze(0))[0]
            s_tf, _ = rssm.observe(s_tf, acts[t].unsqueeze(0), et)
        tf_preds = torch.stack(tf_preds)

    reals = imgs[start + 1:start + 1 + steps].float()

    # ---- 维度 / 数值健全性检查 ----
    assert len(states) == steps and len(infos) == steps, "状态序列长度错误"
    assert all(x.det.shape == (1, M["det"]) for x in states), "确定性态维度错误"
    assert all(x.stoch.shape == (1, M["stoch"]) for x in states), "随机态维度错误"
    assert all(x.stoch.min() >= -1 and x.stoch.max() <= 1 for x in states), "随机态越界"
    assert all(torch.isfinite(x.det).all() and torch.isfinite(x.stoch).all() for x in states)
    assert open_preds.shape == (steps, 3, 64, 64), open_preds.shape
    assert torch.isfinite(open_preds).all(), "预测图像出现 NaN/Inf"
    print(f"[OPEN] 维度检查通过: {steps} 步状态 det=(1,{M['det']}) "
          f"stoch=(1,{M['stoch']})，解码图像 {tuple(open_preds.shape)}", flush=True)

    # ---- 逐步 PSNR / SSIM ----
    def per_step(pred_seq):
        rec = []
        for k in range(steps):
            p = pred_seq[k].float().clamp(-1, 1)
            t = reals[k].clamp(-1, 1)
            mse = F.mse_loss(p, t).item()
            psnr = 10.0 * math.log10(4.0 / (mse + 1e-8))
            a = p.cpu().numpy().transpose(1, 2, 0)
            b = t.cpu().numpy().transpose(1, 2, 0)
            ss = ssim(b, a, channel_axis=2, data_range=2.0)
            rec.append({"step": k + 1, "psnr": round(psnr, 3), "ssim": round(float(ss), 4)})
        return rec

    open_rec, tf_rec = per_step(open_preds), per_step(tf_preds)
    open_psnr = [r["psnr"] for r in open_rec]
    tf_psnr = [r["psnr"] for r in tf_rec]
    head3, tail3 = open_psnr[:3], open_psnr[-3:]
    summary = {
        "open_psnr_mean": round(float(np.mean(open_psnr)), 3),
        "open_psnr_first3": round(float(np.mean(head3)), 3),
        "open_psnr_last3": round(float(np.mean(tail3)), 3),
        "open_psnr_drop": round(float(np.mean(head3) - np.mean(tail3)), 3),
        "tf_psnr_mean": round(float(np.mean(tf_psnr)), 3),
    }
    print(f"[OPEN] 开环 PSNR: 前3步 {summary['open_psnr_first3']:.2f}dB → "
          f"后3步 {summary['open_psnr_last3']:.2f}dB（累积下降 "
          f"{summary['open_psnr_drop']:.2f}dB）；teacher-forcing {summary['tf_psnr_mean']:.2f}dB",
          flush=True)

    out_dir = args.out_dir
    vis_dir = os.path.join(out_dir, "vis")
    met_dir = os.path.join(out_dir, "metrics")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    metrics = {"ckpt": ckpt_path, "traj": args.traj, "start": start, "steps": steps,
               "open_loop": open_rec, "teacher_forcing": tf_rec, "summary": summary}
    with open(os.path.join(met_dir, "open_loop_eval.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 3xN 静态对比：真实 / 开环 / 单步(teacher forcing)
    grid = make_grid(torch.cat([reals, open_preds, tf_preds], dim=0), nrow=steps,
                     normalize=True, value_range=(-1, 1), pad_value=0.4)
    png_path = os.path.join(vis_dir, "open_loop_rollout.png")
    save_image(grid, png_path)
    print(f"[OPEN] 静态对比网格: {png_path}", flush=True)

    # 动图：每帧 [真实 | 开环预测]
    panel, bar = 192, 34
    gif_frames = []
    for k in range(steps):
        canvas = Image.new("RGB", (panel * 2 + 12, panel + bar), (32, 32, 38))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), f"t={k + 1:02d}  REAL", fill=(235, 235, 235))
        draw.text((panel + 20, 10),
                  f"OPEN-LOOP k={k + 1}  PSNR={open_rec[k]['psnr']:.1f}dB",
                  fill=(255, 214, 102))
        canvas.paste(to_pil(reals[k], panel), (0, bar))
        canvas.paste(to_pil(open_preds[k], panel), (panel + 12, bar))
        gif_frames.append(canvas)
    gif_path = os.path.join(vis_dir, "open_loop_rollout.gif")
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=700, loop=0, disposal=2)
    print(f"[OPEN] 动图: {gif_path} ({len(gif_frames)} 帧)", flush=True)
    print("[OPEN] OK: 开环多步预测函数可用，未出现维度错误。", flush=True)


if __name__ == "__main__":
    main()
