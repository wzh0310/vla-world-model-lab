#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""世界模型联合训练：重建损失 + KL 散度（free bits）+ 奖励预测损失。

数据流（20 步序列，单步监督）：
  每步 t 用真实图像 x_t 经编码器得 embed_t，调用 RSSM.observe 得后验状态；
  后验特征 -> 解码器重建 x_t（重建损失）、-> 奖励头预测 r_t（奖励损失）；
  同一 observe 返回的 prior/post 分布参数算 KL（free bits 防初期坍缩）。
  推理（imagine_step）不参与训练损失，留待后续 rollout。

用法（vla 环境）：
  python trainer/train_world_model.py --config configs/default.yaml   # 从配置读超参
  python trainer/train_world_model.py --max-steps 3000 --batch-size 32
  python trainer/train_world_model.py --config configs/default.yaml --max-steps 5000
  python trainer/train_world_model.py --max-steps 3000 --resume outputs/wm/checkpoints/wm_step001000.pt

注意：--config 只填充命令行未显式给出的参数，显式参数始终优先。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, ROOT)
from fast_loader import GPUSequenceSampler
from models.vae import ConvEncoder, ConvDecoder
from models.rssm import RSSM


def parse_args():
    ap = argparse.ArgumentParser(description="世界模型联合训练")
    ap.add_argument("--config", default=None,
                    help="可选 YAML 配置（configs/default.yaml）；显式命令行参数优先")
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "outputs/wm"))
    ap.add_argument("--latent-dim", type=int, default=64, help="编码器 embed 与 AE 隐变量维度")
    ap.add_argument("--stoch-dim", type=int, default=64)
    ap.add_argument("--det-dim", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--free-bits", type=float, default=1.0, help="KL 每元素下限(nats)，防坍缩")
    ap.add_argument("--kl-scale", type=float, default=1.0)
    ap.add_argument("--reward-scale", type=float, default=2.0)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--vis-interval", type=int, default=500)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    if args.config:
        apply_config(args, sys.argv[1:])
    return args


def apply_config(args, argv):
    """用 YAML 配置填充未在命令行显式给出的参数（显式参数优先）。

    配置来源为 configs/default.yaml 的 data / model.rssm / training.world_model 三段。
    这样既集中管理超参数，又不破坏已有命令行复现方式。
    """
    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("data", {})
    m = cfg.get("model", {})
    r = m.get("rssm", {})
    t = cfg.get("training", {}).get("world_model", {})

    # argparse 参数名 -> 配置值
    mapping = {
        "data": os.path.join(ROOT, d["hdf5_path"]) if d.get("hdf5_path") else args.data,
        "seq_len": d.get("seq_len"), "batch_size": d.get("batch_size"), "seed": d.get("seed"),
        "latent_dim": m.get("latent_dim"),
        "stoch_dim": r.get("stoch_dim"), "det_dim": r.get("det_dim"),
        "lr": t.get("lr"), "weight_decay": t.get("weight_decay"),
        "max_steps": t.get("max_steps"), "warmup_steps": t.get("warmup_steps"),
        "grad_clip": t.get("grad_clip"), "free_bits": t.get("free_bits"),
        "kl_scale": t.get("kl_scale"), "reward_scale": t.get("reward_scale"),
        "log_interval": t.get("log_interval"), "vis_interval": t.get("vis_interval"),
        "save_interval": t.get("save_interval"),
    }
    # 命令行显式指定的参数名（--lr -> lr）
    explicit = {tok[2:].replace("-", "_") for tok in argv if tok.startswith("--")}
    for key, value in mapping.items():
        if value is not None and key not in explicit:
            setattr(args, key, value)
    print(f"[CFG] 已加载配置 {args.config}（命令行显式参数优先）", flush=True)


def kl_gaussian(pm, pl, qm, ql, min_std=0.1):
    """KL(q||p)，q=后验, p=先验，均为对角高斯（softplus 参数化方差）。"""
    pv = F.softplus(pl) + min_std
    qv = F.softplus(ql) + min_std
    kl = 0.5 * (torch.log(pv / qv) + (qv + (qm - pm) ** 2) / pv - 1.0)
    return kl  # (B, stoch_dim)


class WorldModel(nn.Module):
    """编码器(复用AE) + RSSM + 解码器(264->图像) + 奖励头。"""

    def __init__(self, latent_dim, stoch_dim, det_dim, ae_ckpt, device):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        feat_dim = det_dim + stoch_dim
        self.decoder = ConvDecoder(latent_dim=feat_dim)
        self.rssm = RSSM(stoch_dim=stoch_dim, det_dim=det_dim,
                         action_dim=4, embed_dim=latent_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        # 从 AE 检查点迁移权重：encoder 全部 + decoder 的 deconv 部分
        state = torch.load(ae_ckpt, map_location=device, weights_only=False)
        ae_dim = state.get('latent_dim', latent_dim)
        if ae_dim != latent_dim:
            raise ValueError(f"AE 隐维度 {ae_dim} 与 --latent-dim {latent_dim} 不一致")
        enc_sd = {k[len('encoder.'):]: v for k, v in state['model'].items() if k.startswith('encoder.')}
        self.encoder.load_state_dict(enc_sd)
        dec_sd = {k[len('decoder.deconv.'):]: v for k, v in state['model'].items()
                  if k.startswith('decoder.deconv.')}
        self.decoder.deconv.load_state_dict(dec_sd)
        print(f"[WM] 已从 AE 迁移 encoder 与 decoder.deconv 权重（AE step={state.get('step')}）", flush=True)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def lr_at_step(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    vis_dir = os.path.join(args.output_dir, 'vis')
    log_dir = os.path.join(args.output_dir, 'logs')
    for d in (ckpt_dir, vis_dir, log_dir):
        os.makedirs(d, exist_ok=True)
    writer = SummaryWriter(log_dir)

    sampler = GPUSequenceSampler(args.data, seq_len=args.seq_len, image_size=64,
                                 device=args.device, seed=args.seed)
    print(f"[WM] 采样器: {len(sampler)} 序列, 图像显存 {sampler.mem_mb:.1f}MB, "
          f"加载 {sampler.load_sec:.1f}s", flush=True)

    model = WorldModel(args.latent_dim, args.stoch_dim, args.det_dim,
                       args.ae_ckpt, args.device).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[WM] 参数量={model.count_parameters():,} 设备={args.device} "
          f"max_steps={args.max_steps} free_bits={args.free_bits}", flush=True)

    start = 0
    if args.resume:
        st = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['optimizer'])
        start = st.get('step', 0)
        print(f"[WM] 续训 {args.resume} step={start}", flush=True)

    model.train()
    t0 = time.time()
    step = start
    acc = {'rec': 0.0, 'kl': 0.0, 'rew': 0.0, 'n': 0}
    while step < args.max_steps:
        imgs, acts, rews = sampler.sample(args.batch_size, with_rewards=True)  # (B,T,...)
        B, T = imgs.shape[:2]
        lr = lr_at_step(step, args.lr, args.warmup_steps, args.max_steps)
        for g in opt.param_groups:
            g['lr'] = lr

        # 编码所有帧（单步监督：每步都用真实图像）
        flat_x = imgs.reshape(B * T, 3, 64, 64)
        embeds, _ = model.encoder(flat_x)
        embeds = embeds.view(B, T, -1)

        state = model.rssm.initial_state(B, device=args.device)
        total_rec = total_kl = total_rew = 0.0
        for t in range(T):
            state, info = model.rssm.observe(state, acts[:, t], embeds[:, t])
            feat = model.rssm.get_feature(state)              # (B,264)
            recon = model.decoder(feat)                       # (B,3,64,64)
            rec_t = F.mse_loss(recon, imgs[:, t])
            kl_t = kl_gaussian(info['prior_mu'], info['prior_logvar'],
                               info['post_mu'], info['post_logvar'], model.rssm.min_std)
            # free bits: 每个样本的元素级 KL 下限（先对元素维取均值得到每行 KL）
            kl_t = kl_t.mean(dim=-1).clamp_min(args.free_bits).mean()
            pred_r = model.reward_head(feat).squeeze(-1)      # (B,)
            rew_t = F.mse_loss(pred_r, rews[:, t].squeeze(-1))
            total_rec += rec_t
            total_kl += kl_t
            total_rew += rew_t

        loss_rec = total_rec / T
        loss_kl = total_kl / T
        loss_rew = total_rew / T
        loss = loss_rec + args.kl_scale * loss_kl + args.reward_scale * loss_rew

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1

        acc['rec'] += loss_rec.item(); acc['kl'] += loss_kl.item()
        acc['rew'] += loss_rew.item(); acc['n'] += 1

        if step % args.log_interval == 0 or step == 1:
            n = acc['n']
            print(f"[WM] step {step}/{args.max_steps} rec={acc['rec']/n:.5f} "
                  f"kl={acc['kl']/n:.4f} rew={acc['rew']/n:.5f} "
                  f"loss={loss.item():.4f} gnorm={grad_norm.item():.3f} "
                  f"lr={lr:.2e} {time.time()-t0:.0f}s", flush=True)
            writer.add_scalar('loss/recon', acc['rec']/n, step)
            writer.add_scalar('loss/kl', acc['kl']/n, step)
            writer.add_scalar('loss/reward', acc['rew']/n, step)
            writer.add_scalar('loss/total', loss.item(), step)
            writer.add_scalar('train/grad_norm', grad_norm.item(), step)
            writer.add_scalar('train/lr', lr, step)
            acc = {'rec': 0.0, 'kl': 0.0, 'rew': 0.0, 'n': 0}

        if step % args.vis_interval == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                vi, va, vr = sampler.sample(8, with_rewards=True)
                VB = vi.shape[0]
                ve, _ = model.encoder(vi.reshape(VB*T, 3, 64, 64))
                ve = ve.view(VB, T, -1)
                st = model.rssm.initial_state(VB, device=args.device)
                for t in range(T):
                    st, _ = model.rssm.observe(st, va[:, t], ve[:, t])
                vrec = model.decoder(model.rssm.get_feature(st))
                grid = make_grid(torch.cat([vi[:, T//2], vrec], dim=0),
                                 nrow=8, normalize=True, value_range=(-1, 1), pad_value=0.3)
                path = os.path.join(vis_dir, f'wm_recon_step{step:06d}.png')
                save_image(grid, path)
                writer.add_image('wm/recon', grid, step)
            model.train()
            print(f"[WM] 重建可视化: {path}", flush=True)

        if step % args.save_interval == 0 or step == args.max_steps:
            path = os.path.join(ckpt_dir, f'wm_step{step:06d}.pt')
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                        'step': step, 'args': vars(args)}, path)
            print(f"[WM] 检查点: {path}", flush=True)

    writer.close()
    print(f"[WM] 训练完成 {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
