#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""视觉自编码器训练脚本（MSE 重建损失）。

功能：学习率余弦衰减 + warmup、检查点保存（含完整训练状态可续训）、
TensorBoard 日志、每 500 步保存「原图 - 重建图」对比图到 outputs/ae_vis。

用法（vla 环境）：
  python trainer/train_ae.py --max-steps 100 --batch-size 16          # 试训
  python trainer/train_ae.py --config configs/world_model.yaml         # 完整训练
  python trainer/train_ae.py --resume outputs/checkpoints/ae_step2000.pt
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, ROOT)
from episodic_dataset import EpisodicDataset
from fast_loader import GPUSequenceSampler
from models.vae import ConvVAE


def parse_args():
    ap = argparse.ArgumentParser(description="视觉自编码器训练")
    ap.add_argument("--config", default=None, help="可选 YAML 配置（需 pyyaml）")
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--vis-interval", type=int, default=500)
    ap.add_argument("--save-interval", type=int, default=2000)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--num-vis", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None, help="检查点路径，续训")
    ap.add_argument("--fast", dest="fast", action="store_true", default=True,
                    help="GPU 内存预加载采样（默认开启，约 80x 加速）")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="回退到 DataLoader + EpisodicDataset（慢）")
    args = ap.parse_args()
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        d, m, t, e = cfg.get("data", {}), cfg.get("model", {}), cfg.get("training", {}), cfg.get("eval", {})
        args.data = os.path.join(ROOT, d.get("hdf5_path", args.data))
        args.seq_len, args.image_size = d.get("seq_len", args.seq_len), d.get("image_size", args.image_size)
        args.batch_size, args.num_workers = d.get("batch_size", args.batch_size), d.get("num_workers", args.num_workers)
        args.latent_dim = m.get("latent_dim", args.latent_dim)
        args.max_steps, args.lr = t.get("total_steps", args.max_steps), t.get("lr", args.lr)
        args.log_interval, args.save_interval = t.get("log_interval", args.log_interval), t.get("save_interval", args.save_interval)
        args.device = t.get("device", args.device)
        args.num_vis = e.get("log_images", args.num_vis)
    return args


def lr_at_step(step, base_lr, warmup, total):
    """线性 warmup + 余弦衰减到 0。"""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def run_eval(model, loader, device, max_batches=10):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch["image_seq"].to(device, non_blocking=True)
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            recon = model(x)["recon"]
            losses.append(F.mse_loss(recon, x).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退 CPU")
        args.device = "cpu"

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    vis_dir = os.path.join(args.output_dir, "ae_vis")
    log_dir = os.path.join(args.output_dir, "logs", "ae")
    for d in (ckpt_dir, vis_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    if args.fast:
        sampler = GPUSequenceSampler(args.data, seq_len=args.seq_len,
                                     image_size=args.image_size,
                                     device=args.device, seed=args.seed)
        loader = None
        dataset = sampler
        print(f"[AE] GPU 预加载: {len(sampler)} 序列, 图像显存 {sampler.mem_mb:.1f}MB, "
              f"加载 {sampler.load_sec:.1f}s", flush=True)
        # 固定验证帧：从采样器取一批后取前 16 帧
        val_x = sampler.sample(min(16, args.batch_size))[0][:, :1]  # (B,1,3,H,W)
    else:
        dataset = EpisodicDataset(args.data, seq_len=args.seq_len, image_size=args.image_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
                            drop_last=True, persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_gen = iter(val_loader)
        val_batches = [next(val_gen)["image_seq"][:8] for _ in range(2)]
        val_x = torch.cat(val_batches, dim=0)

    model = ConvVAE(latent_dim=args.latent_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir)
    print(f"[AE] 序列数={len(dataset)} 参数量={model.count_parameters():,} "
          f"设备={args.device} 隐变量={args.latent_dim} 最大步数={args.max_steps}", flush=True)

    start_step, step, t0 = 0, 0, time.time()
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state.get("step", 0)
        step = start_step
        print(f"[AE] 从 {args.resume} 续训（step={start_step}）", flush=True)

    model.train()
    running_loss, running_count = 0.0, 0
    done = False
    while not done:
        if args.fast:
            x, _ = sampler.sample(args.batch_size)       # 已在显存
        else:
            batch = next(iter(loader))
            x = batch["image_seq"].to(args.device, non_blocking=True)
        if x is None:
            break
        if True:
            b, t, c, h, w = x.shape
            flat = x.reshape(b * t, c, h, w)
            lr = lr_at_step(step, args.lr, args.warmup_steps, args.max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            recon = model(flat)["recon"]
            loss = F.mse_loss(recon, flat)  # 纯 MSE 重建损失
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item() * flat.shape[0]
            running_count += flat.shape[0]
            step += 1

            if step % args.log_interval == 0 or step == 1:
                avg = running_loss / max(1, running_count)
                writer.add_scalar("train/mse", avg, step)
                writer.add_scalar("train/lr", lr, step)
                print(f"[AE] step {step}/{args.max_steps}  train_mse={avg:.5f}  "
                      f"lr={lr:.2e}  {time.time()-t0:.0f}s", flush=True)
                running_loss, running_count = 0.0, 0

            if step % args.vis_interval == 0 or step == args.max_steps:
                with torch.no_grad():
                    model.eval()
                    vis_seq = val_x[:1].to(args.device)            # (1,T,C,H,W)
                    b, t, c, h, w = vis_seq.shape
                    vis_x = vis_seq.reshape(b * t, c, h, w)[:args.num_vis]  # 展平为单帧
                    vis_r = model(vis_x)["recon"]
                    # 上半原图、下半重建
                    grid = make_grid(torch.cat([vis_x, vis_r], dim=0), nrow=args.num_vis,
                                     normalize=True, value_range=(-1, 1), pad_value=0.3)
                    path = os.path.join(vis_dir, f"recon_step{step:06d}.png")
                    save_image(grid, path)
                    writer.add_image("重建对比", grid, step)
                    model.train()
                print(f"[AE] 可视化已保存: {path}", flush=True)

            if args.eval_interval > 0 and step % args.eval_interval == 0:
                if args.fast:
                    model.eval()
                    with torch.no_grad():
                        ex = sampler.sample(min(32, args.batch_size))[0]
                        B, T = ex.shape[:2]
                        val_loss = float(F.mse_loss(model(ex.reshape(B*T, 3, args.image_size, args.image_size))["recon"],
                                                    ex.reshape(B*T, 3, args.image_size, args.image_size)))
                    model.train()
                else:
                    val_loss = run_eval(model, val_loader, args.device)
                writer.add_scalar("eval/mse", val_loss, step)
                print(f"[AE] step {step}  eval_mse={val_loss:.5f}", flush=True)

            if step % args.save_interval == 0 or step == args.max_steps:
                path = os.path.join(ckpt_dir, f"ae_step{step:06d}.pt")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "args": vars(args)}, path)
                print(f"[AE] 检查点已保存: {path}", flush=True)

            if step >= args.max_steps:
                done = True
                break

    writer.close()
    print(f"[AE] 训练完成，总耗时 {time.time()-t0:.0f}s（起始 step={start_step}）", flush=True)


if __name__ == "__main__":
    main()
