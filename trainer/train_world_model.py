#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""世界模型联合训练：重建 + KL（free bits）+ 奖励预测 + 价值预测 四损失。

数据流（20 步序列，单步监督）：
  每步 t 用真实图像 x_t 经编码器得 embed_t，调用 RSSM.observe 得后验状态；
  后验特征 -> 解码器重建 x_t（重建损失）、-> 奖励头预测 r_t（奖励损失）、
  -> 价值头预测 V_t（价值损失，目标为该步起的折扣回报）；
  同一 observe 返回的 prior/post 分布参数算 KL（free bits 防初期坍缩）。
  推理（imagine_step / imagine_rollout）不参与训练损失，留待想象策略训练。

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
from models.factory import build_dynamics, build_reward_heads


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
    ap.add_argument("--dynamics", type=str, default="rssm", choices=["rssm", "mlp"],
                    help="动力学类型：rssm（循环+随机）/ mlp（纯前馈基线，消融对照）")
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=64, help="图像分辨率(16 的倍数)")
    ap.add_argument("--kl-warmup", type=int, default=0,
                    help="KL 权重退火步数：>0 时 kl_scale 从 0.1 线性升至 --kl-scale（防训练初期隐状态坍缩）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="统一学习率（分组关闭时的后备值）")
    ap.add_argument("--lr-enc", type=float, default=1e-4, help="编码器+解码器学习率（默认 1e-4）")
    ap.add_argument("--lr-dyn", type=float, default=3e-4, help="RSSM+奖励头+价值头学习率（默认 3e-4）")
    ap.add_argument("--group-lr", action="store_true", help="启用分组学习率（enc 1e-4 / dyn 3e-4）")
    ap.add_argument("--seq-len-start", type=int, default=10,
                    help="课程学习起始序列长度（默认 10，线性增长到 --seq-len=20）")
    ap.add_argument("--curriculum-steps", type=int, default=1000,
                    help="序列长度从 start 线性增到全长所需的步数；0 表示关闭课程")
    ap.add_argument("--eval-interval", type=int, default=1000,
                    help="每多少步在验证集上评估一次并保存最优权重；0 关闭")
    ap.add_argument("--eval-data", default=None,
                    help="验证集 HDF5（ws_dataset/test.h5）；缺省则复用训练数据同轨迹")
    ap.add_argument("--eval-trajs", type=int, default=6, help="评估用轨迹数")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--free-bits", type=float, default=1.0, help="KL 每元素下限(nats)，防坍缩")
    ap.add_argument("--kl-scale", type=float, default=1.0)
    ap.add_argument("--reward-scale", type=float, default=2.0)
    ap.add_argument("--value-scale", type=float, default=1.0, help="价值损失权重")
    ap.add_argument("--gamma", type=float, default=0.99, help="价值目标折扣因子")
    ap.add_argument("--augment", action="store_true", help="开启图像增强（小幅平移+亮度扰动）")
    ap.add_argument("--shift-px", type=float, default=2.0, help="增强平移幅度(像素)")
    ap.add_argument("--brightness", type=float, default=0.1, help="增强亮度扰动幅度")
    ap.add_argument("--crop-frac", type=float, default=0.08, help="增强随机裁剪幅度(比例)")
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
        "seq_len": d.get("seq_len"), "image_size": d.get("image_size"), "batch_size": d.get("batch_size"), "seed": d.get("seed"),
        "latent_dim": m.get("latent_dim"),
        "dynamics": m.get("dynamics"),
        "stoch_dim": r.get("stoch_dim"), "det_dim": r.get("det_dim"),
        "lr": t.get("lr"), "weight_decay": t.get("weight_decay"),
        "max_steps": t.get("max_steps"), "warmup_steps": t.get("warmup_steps"),
        "grad_clip": t.get("grad_clip"), "free_bits": t.get("free_bits"),
        "kl_scale": t.get("kl_scale"), "kl_warmup": t.get("kl_warmup"), "reward_scale": t.get("reward_scale"),
        "value_scale": t.get("value_scale"), "gamma": t.get("gamma"),
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


def discounted_returns(rewards, gamma):
    """从每步起的折扣回报，作为价值头的监督目标。

    rewards: (B, T, 1) 归一化奖励（r/10 裁剪 [-1,1]）
    返回:    (B, T)   G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    采用有限窗口（末端不 bootstrap），因为当前数据为定长 20 步序列。
    """
    B, T, _ = rewards.shape
    r = rewards.squeeze(-1)                       # (B,T)
    G = torch.zeros_like(r)
    running = torch.zeros(B, device=r.device, dtype=r.dtype)
    for t in reversed(range(T)):
        running = r[:, t] + gamma * running
        G[:, t] = running
    return G


class WorldModel(nn.Module):
    """编码器(复用AE) + 动力学(RSSM 或 MLP 基线) + 解码器 + 奖励头 + 价值头。

    dynamics 参数由模型工厂构建，属性名统一为 self.rssm，使训练/评估/推理
    对两种动力学使用完全相同的调用路径（消融时仅改一行配置）。
    """

    def __init__(self, latent_dim, stoch_dim, det_dim, ae_ckpt, device, image_size=64,
                 dynamics="rssm"):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        feat_dim = det_dim + stoch_dim
        self.decoder = ConvDecoder(latent_dim=feat_dim, out_size=image_size)
        self.rssm, _ = build_dynamics(dynamics, feat_dim, 4, latent_dim,
                                      stoch_dim, det_dim)
        self.dynamics = dynamics
        # 两个头继承 nn.Sequential，权重 key 与旧内联 reward_head 一致（便于兼容历史检查点）
        self.reward_head, self.value_head = build_reward_heads(feat_dim)
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


def kl_scale_at_step(step, target, warmup):
    """KL 权重退火：前 warmup 步从 0.1 线性升至 target，之后恒定 target。"""
    if warmup <= 0 or step >= warmup:
        return target
    return 0.1 + (target - 0.1) * step / max(1, warmup)


def lr_at_step(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def curriculum_len(step, start_len, full_len, ramp_steps):
    """课程学习：前 ramp_steps 步把序列长度从 start_len 线性增到 full_len。"""
    if ramp_steps <= 0 or step >= ramp_steps:
        return full_len
    return max(2, int(round(start_len + (full_len - start_len) * step / ramp_steps)))


@torch.no_grad()
def evaluate_openloop(model, eval_sampler, image_size, steps=20, n_traj=6,
                      device="cuda", seed=1234):
    """在验证集上跑开环 20 步预测，返回 (ssim_last20, reward_mae, psnr_last20)。

    口径与 scripts/eval_world_model.py 一致：首帧后验作起点 + 真实动作开环想象，
    PSNR/SSIM 用全部 n_traj×steps 步平均，reward_mae 为归一化尺度 MAE。
    """
    from trainer.metrics import psnr_metric, ssim_metric
    model.eval()
    rng = np.random.default_rng(seed)
    # 取 n_traj 个互不相同的序列起点
    n = min(n_traj, len(eval_sampler.index))
    pick = rng.choice(len(eval_sampler.index), size=n, replace=False)
    psnrs, ssims, rew_errs = [], [], []
    for idx in pick:
        tid, s = eval_sampler.index[idx]
        tid, s = int(tid), int(s)
        seq_len = min(steps + 1, len(eval_sampler.images[tid]) - s)
        imgs = eval_sampler.images[tid][s:s + seq_len].float()      # (L,3,S,S)
        acts = eval_sampler.actions[tid][s:s + seq_len]             # (L,4)
        rews = eval_sampler.rewards[tid][s:s + seq_len]             # (L,1)
        if seq_len < steps + 1:
            continue
        # 首帧后验起点
        emb0, _ = model.encoder(imgs[0:1].unsqueeze(0).reshape(1, 3, image_size, image_size))
        st = model.rssm.initial_state(1, device=device)
        st, _ = model.rssm.observe(st, acts[0:1].to(device), emb0.view(1, -1))
        # 开环想象 steps 步（真实动作 t=1..steps）
        act_seq = acts[1:steps + 1].unsqueeze(0).to(device)
        states, _ = model.rssm.imagine_rollout(st, act_seq, sample=False)
        preds = model.rssm.decode_rollout(states, model.decoder).clamp(-1, 1)[:, 0].cpu()
        reals = imgs[1:steps + 1].cpu()
        psnrs.append(psnr_metric(preds, reals).mean())
        ssims.append(ssim_metric(preds, reals).mean())
        feat = torch.stack([model.rssm.get_feature(x).reshape(-1) for x in states])
        pred_r = model.reward_head(feat.to(device)).squeeze(-1).cpu()
        rew_errs.append((pred_r - rews[1:steps + 1].squeeze(-1).cpu()).abs().mean())
    model.train()
    if not ssims:
        return 0.0, 999.0, 0.0
    return float(np.mean(ssims)), float(np.mean(rew_errs)), float(np.mean(psnrs))


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

    sampler = GPUSequenceSampler(args.data, seq_len=args.seq_len, image_size=args.image_size,
                                 device=args.device, seed=args.seed,
                                 augment=args.augment, shift_px=args.shift_px,
                                 brightness=args.brightness, crop_frac=args.crop_frac)
    print(f"[WM] 采样器: {len(sampler)} 序列, 图像显存 {sampler.mem_mb:.1f}MB, "
          f"加载 {sampler.load_sec:.1f}s", flush=True)

    model = WorldModel(args.latent_dim, args.stoch_dim, args.det_dim,
                       args.ae_ckpt, args.device, image_size=args.image_size,
                       dynamics=args.dynamics).to(args.device)
    if args.group_lr:
        enc_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        dyn_params = (list(model.rssm.parameters()) + list(model.reward_head.parameters())
                      + list(model.value_head.parameters()))
        opt = torch.optim.AdamW([
            {"params": enc_params, "lr": args.lr_enc},
            {"params": dyn_params, "lr": args.lr_dyn},
        ], weight_decay=args.weight_decay)
        print(f"[WM] 分组学习率: enc={args.lr_enc} (encoder+decoder) "
              f"dyn={args.lr_dyn} (RSSM+heads)", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[WM] 参数量={model.count_parameters():,} 设备={args.device} "
          f"max_steps={args.max_steps} free_bits={args.free_bits} kl_warmup={args.kl_warmup}", flush=True)

    # 验证集采样器（不增强），用于周期性开环评估
    eval_sampler = None
    if args.eval_interval > 0:
        eval_path = args.eval_data if args.eval_data else args.data
        if os.path.isfile(eval_path):
            eval_sampler = GPUSequenceSampler(eval_path, seq_len=args.seq_len,
                                              image_size=args.image_size,
                                              device=args.device, seed=args.seed + 999,
                                              augment=False)
            print(f"[WM] 验证集: {eval_path} ({len(eval_sampler)} 序列)", flush=True)
        else:
            print(f"[WM] 警告: 验证集 {eval_path} 不存在，周期性评估关闭", flush=True)
            args.eval_interval = 0

    best_ssim = -1.0
    best_step = -1

    start = 0
    if args.resume:
        st = torch.load(args.resume, map_location=args.device, weights_only=False)
        # 兼容旧检查点（无 value_head）：缺失的 value_head 随机初始化，不强制 strict
        missing, unexpected = model.load_state_dict(st['model'], strict=False)
        if missing:
            print(f"[WM] 续训缺失权重(将随机初始化): {missing}", flush=True)
        if unexpected:
            print(f"[WM] 续训多余权重(已忽略): {unexpected}", flush=True)
        # optimizer 状态含旧参数形状时可能不匹配，尝试加载否则重建
        try:
            opt.load_state_dict(st['optimizer'])
        except ValueError as e:
            print(f"[WM] optimizer 状态不兼容，重新初始化: {e}", flush=True)
        start = st.get('step', 0)
        print(f"[WM] 续训 {args.resume} step={start}", flush=True)

    model.train()
    t0 = time.time()
    step = start
    acc = {'rec': 0.0, 'kl': 0.0, 'rew': 0.0, 'val': 0.0, 'n': 0}
    while step < args.max_steps:
        # 课程学习：序列长度从 seq_len_start 线性增到 seq_len
        cur_len = curriculum_len(step, args.seq_len_start, args.seq_len, args.curriculum_steps)
        imgs, acts, rews = sampler.sample(args.batch_size, with_rewards=True,
                                          seq_len=cur_len)  # (B,T,...)
        B, T = imgs.shape[:2]
        # 价值目标：每步起的折扣回报（归一化尺度，与奖励同源）
        value_target = discounted_returns(rews, args.gamma)   # (B,T)
        lr = lr_at_step(step, args.lr, args.warmup_steps, args.max_steps)
        kl_scale_now = kl_scale_at_step(step, args.kl_scale, args.kl_warmup)
        if args.group_lr:
            lr_e = lr_at_step(step, args.lr_enc, args.warmup_steps, args.max_steps)
            lr_d = lr_at_step(step, args.lr_dyn, args.warmup_steps, args.max_steps)
            opt.param_groups[0]['lr'] = lr_e
            opt.param_groups[1]['lr'] = lr_d
        else:
            for g in opt.param_groups:
                g['lr'] = lr

        # 编码所有帧（单步监督：每步都用真实图像）
        flat_x = imgs.reshape(B * T, 3, args.image_size, args.image_size)
        embeds, _ = model.encoder(flat_x)
        embeds = embeds.view(B, T, -1)

        state = model.rssm.initial_state(B, device=args.device)
        total_rec = total_kl = total_rew = total_val = 0.0
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
            pred_v = model.value_head(feat).squeeze(-1)       # (B,)
            val_t = F.mse_loss(pred_v, value_target[:, t])
            total_rec += rec_t
            total_kl += kl_t
            total_rew += rew_t
            total_val += val_t

        loss_rec = total_rec / T
        loss_kl = total_kl / T
        loss_rew = total_rew / T
        loss_val = total_val / T
        loss = (loss_rec + kl_scale_now * loss_kl
                + args.reward_scale * loss_rew + args.value_scale * loss_val)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1

        acc['rec'] += loss_rec.item(); acc['kl'] += loss_kl.item()
        acc['rew'] += loss_rew.item(); acc['val'] += loss_val.item(); acc['n'] += 1

        if step % args.log_interval == 0 or step == 1:
            n = acc['n']
            print(f"[WM] step {step}/{args.max_steps} rec={acc['rec']/n:.5f} "
                  f"kl={acc['kl']/n:.4f} rew={acc['rew']/n:.5f} val={acc['val']/n:.5f} "
                  f"loss={loss.item():.4f} gnorm={grad_norm.item():.3f} "
                  f"lr={lr:.2e} ks={kl_scale_now:.3f} {time.time()-t0:.0f}s", flush=True)
            writer.add_scalar('loss/recon', acc['rec']/n, step)
            writer.add_scalar('loss/kl', acc['kl']/n, step)
            writer.add_scalar('loss/reward', acc['rew']/n, step)
            writer.add_scalar('loss/value', acc['val']/n, step)
            writer.add_scalar('loss/total', loss.item(), step)
            writer.add_scalar('train/grad_norm', grad_norm.item(), step)
            writer.add_scalar('train/lr', lr, step)
            writer.add_scalar('train/kl_scale', kl_scale_now, step)
            acc = {'rec': 0.0, 'kl': 0.0, 'rew': 0.0, 'val': 0.0, 'n': 0}

        if step % args.vis_interval == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                vi, va, vr = sampler.sample(8, with_rewards=True)
                VB, VT = vi.shape[0], vi.shape[1]   # 用采样自身的长度，避免与课程长度 T 不一致
                ve, _ = model.encoder(vi.reshape(VB*VT, 3, args.image_size, args.image_size))
                ve = ve.view(VB, VT, -1)
                st = model.rssm.initial_state(VB, device=args.device)
                for t in range(VT):
                    st, _ = model.rssm.observe(st, va[:, t], ve[:, t])
                vrec = model.decoder(model.rssm.get_feature(st))
                grid = make_grid(torch.cat([vi[:, VT//2], vrec], dim=0),
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

        # 周期性评估（开环 20 步）+ 保存最优权重
        if eval_sampler is not None and (step % args.eval_interval == 0 or step == args.max_steps):
            ev_ssim, ev_rew, ev_psnr = evaluate_openloop(
                model, eval_sampler, args.image_size, steps=args.seq_len,
                n_traj=args.eval_trajs, device=args.device)
            writer.add_scalar('eval/ssim20', ev_ssim, step)
            writer.add_scalar('eval/reward_mae', ev_rew, step)
            writer.add_scalar('eval/psnr20', ev_psnr, step)
            tag = ""
            if ev_ssim > best_ssim:
                best_ssim, best_step = ev_ssim, step
                best_path = os.path.join(ckpt_dir, 'wm_best.pt')
                torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                            'step': step, 'args': vars(args),
                            'eval': {'ssim20': ev_ssim, 'reward_mae': ev_rew, 'psnr20': ev_psnr}},
                           best_path)
                tag = " ★新最优已保存 wm_best.pt"
            print(f"[EVAL] step {step} 开环{args.seq_len}步: SSIM={ev_ssim:.4f} "
                  f"PSNR={ev_psnr:.3f}dB reward_MAE={ev_rew:.5f} "
                  f"(best SSIM={best_ssim:.4f}@{best_step}){tag}", flush=True)

    writer.close()
    print(f"[WM] 训练完成 {time.time()-t0:.0f}s | 最优 SSIM20={best_ssim:.4f}@{best_step}", flush=True)


if __name__ == '__main__':
    main()
