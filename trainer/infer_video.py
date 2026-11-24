#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键推理：输入动作序列 → 开环多步预测 → 生成预测视频（GIF）与对比图。

支持三种动作来源：
  1. --traj + --start：从 HDF5 中某条轨迹的指定起点截取真实动作序列（推荐，可同时对齐真实帧对比）；
  2. --actions-json：从 JSON 文件读取 (N,4) 动作序列（归一化到 [-1,1] 的动作）；
  3. --random-actions：随机生成平滑动作序列（无真实帧对比，仅演示预测视频生成流程）。

流程：真实首帧编码后验作起点 → RSSM.imagine_rollout 纯动力学展开 N 步 → 解码器出图 →
      组装「真实 vs 预测」并排 GIF + 静态网格 + 单帧 PNG。

用法（vla 环境）：
  # 从轨迹推理（含真实对比）
  python trainer/infer_video.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt \
      --traj traj_0002 --start 8 --steps 20 --out outputs/infer_demo
  # 自定义动作序列
  python trainer/infer_video.py --ckpt ... --actions-json my_actions.json --steps 15 --out ...
  # 随机动作演示
  python trainer/infer_video.py --ckpt ... --random-actions --steps 20 --out ...
"""
import argparse
import json
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
sys.path.insert(0, os.path.join(ROOT, "trainer"))
from eval_single_step import build_models
from make_prediction_gif import to_pil


def parse_args():
    ap = argparse.ArgumentParser(description="一键推理：动作序列 → 预测视频")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/wm_tune1/checkpoints/wm_step004000.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/infer_video"))
    ap.add_argument("--steps", type=int, default=20, help="开环预测步数")
    ap.add_argument("--traj", default=None, help="HDF5 轨迹 key，如 traj_0002")
    ap.add_argument("--start", type=int, default=2, help="轨迹起点帧索引")
    ap.add_argument("--actions-json", default=None, help="JSON 文件，含 (N,4) 归一化动作")
    ap.add_argument("--random-actions", action="store_true", help="随机平滑动作（演示用）")
    ap.add_argument("--fps-ms", type=int, default=260, help="GIF 每帧停留毫秒")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_from_hdf5(data_path, traj, start, steps, image_size, device):
    with h5py.File(data_path, "r") as f:
        low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
        high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
        span = torch.where(high == low, torch.ones_like(high), high - low)
        images = f[traj]["images"][:]
        actions = f[traj]["actions"][:]
    raw = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
    small = F.interpolate(raw, size=(image_size, image_size), mode="bilinear", align_corners=False)
    imgs = (small / 127.5 - 1.0).half()                       # (n,3,S,S)
    a_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    acts = ((a_t - low) / span * 2 - 1).clamp(-1, 1)         # (n,4)
    if start + steps >= imgs.shape[0]:
        raise ValueError(f"start({start})+steps({steps}) 超出轨迹长度 {imgs.shape[0]}")
    acts_seq = acts[start + 1:start + 1 + steps].unsqueeze(0)  # (1,N,4)
    init_img = imgs[start].float().unsqueeze(0)                 # (1,3,S,S)
    init_act = acts[start].unsqueeze(0)                         # (1,4)
    reals = imgs[start + 1:start + 1 + steps].float()           # (N,3,S,S)
    return init_img, init_act, acts_seq, reals, traj


def load_from_json(json_path, steps, device):
    with open(json_path, encoding="utf-8") as f:
        obj = json.load(f)
    arr = np.asarray(obj["actions"] if isinstance(obj, dict) else obj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"动作 JSON 应为 (N,4)，实际 {arr.shape}")
    acts = torch.as_tensor(arr, device=device).clamp(-1, 1)
    if acts.shape[0] < steps:
        raise ValueError(f"动作序列长度 {acts.shape[0]} < steps {steps}")
    return acts[:steps].unsqueeze(0)  # (1,N,4)


def random_actions(steps, device, seed):
    rng = np.random.default_rng(seed)
    # 平滑随机游走：每步在上一动作附近扰动，末维 gripper 保持开合随机
    a = np.zeros((steps, 4), dtype=np.float32)
    cur = np.zeros(4, dtype=np.float32)
    for i in range(steps):
        cur[:3] = np.clip(cur[:3] + rng.normal(0, 0.25, 3), -1, 1)
        cur[3] = rng.choice([-1.0, 1.0])
        a[i] = cur
    return torch.as_tensor(a, device=device).unsqueeze(0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    M = build_models(args, ckpt)
    enc, dec, rssm = M["enc"], M["dec"], M["rssm"]

    reals = None
    if args.traj:
        init_img, init_act, acts_seq, reals, tag = load_from_hdf5(
            args.data, args.traj, args.start, args.steps, 64, args.device)
    elif args.actions_json:
        acts_seq = load_from_json(args.actions_json, args.steps, args.device)
        # 无真实首帧：用零图像编码作起点
        init_img = torch.zeros(1, 3, 64, 64, device=args.device)
        init_act = torch.zeros(1, 4, device=args.device)
        tag = "custom"
    elif args.random_actions:
        acts_seq = random_actions(args.steps, args.device, args.seed)
        init_img = torch.zeros(1, 3, 64, 64, device=args.device)
        init_act = torch.zeros(1, 4, device=args.device)
        tag = "random"
    else:
        raise ValueError("请指定 --traj / --actions-json / --random-actions 之一")

    with torch.no_grad():
        e0 = enc(init_img)[0]
        init = rssm.observe(rssm.initial_state(1, device=args.device), init_act, e0)[0]
        states, _ = rssm.imagine_rollout(init, acts_seq)             # 全开环
        preds = rssm.decode_rollout(states, dec)[:, 0].clamp(-1, 1)  # (N,3,S,S)

    steps = args.steps
    # 静态对比网格（前 20 帧）
    n_show = min(steps, 20)
    if reals is not None:
        grid = make_grid(torch.cat([reals[:n_show], preds[:n_show]], dim=0),
                         nrow=n_show, normalize=True, value_range=(-1, 1), pad_value=0.4)
        save_image(grid, os.path.join(args.out, f"{tag}_compare.png"))

    # GIF：真实 vs 预测并排（无真实帧时只显示预测）
    panel, bar = 176, 34
    frames = []
    for k in range(steps):
        dual = reals is not None
        canvas = Image.new("RGB", (panel * (2 if dual else 1) + 12, panel + bar), (32, 32, 38))
        d = ImageDraw.Draw(canvas)
        if dual:
            d.text((8, 9), f"{tag} REAL t={k+1:02d}", fill=(235, 235, 235))
            d.text((panel + 20, 9), f"PRED t={k+1:02d}", fill=(255, 214, 102))
            canvas.paste(to_pil(reals[k], panel), (0, bar))
            canvas.paste(to_pil(preds[k], panel), (panel + 12, bar))
        else:
            d.text((8, 9), f"{tag} PRED t={k+1:02d}", fill=(255, 214, 102))
            canvas.paste(to_pil(preds[k], panel), (0, bar))
        frames.append(canvas)
    gif_path = os.path.join(args.out, f"{tag}_pred.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=args.fps_ms, loop=0, disposal=2)

    # 每帧 PNG
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for k in range(steps):
        save_image(preds[k].float(), os.path.join(frames_dir, f"pred_{k+1:03d}.png"),
                   normalize=True, value_range=(-1, 1))

    info = {"ckpt": args.ckpt, "source": tag, "steps": steps,
            "stoch": M["stoch"], "det": M["det"],
            "gif": gif_path, "frames_dir": frames_dir}
    with open(os.path.join(args.out, "infer_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[INFER] 完成 {steps} 步预测: {gif_path}", flush=True)
    print(f"[INFER] 逐帧 PNG: {frames_dir}", flush=True)


if __name__ == "__main__":
    main()
