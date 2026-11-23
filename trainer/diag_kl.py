#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断后验/先验的真实 KL 分布（不受 free bits clamp 影响），用于设定 free bits。"""
import glob, os, sys
from types import SimpleNamespace
CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/wm/checkpoints/wm_step003000.pt"
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/data/wzh/robot/vla-world-model-lab"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, os.path.join(ROOT, "trainer"))
from fast_loader import GPUSequenceSampler
from eval_single_step import build_models
from train_world_model import kl_gaussian

device = "cuda"
ckpt_path = CKPT if os.path.isabs(CKPT) else os.path.join(ROOT, CKPT)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
args = SimpleNamespace(device=device,
                       ae_ckpt=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
M = build_models(args, ckpt)
sampler = GPUSequenceSampler(os.path.join(ROOT, "outputs/data/trajectories.h5"),
                             seq_len=20, image_size=64, device=device, seed=7)
imgs, acts, rews = sampler.sample(64, with_rewards=True)
B, T = imgs.shape[:2]
kls = []
with torch.no_grad():
    flat = imgs.reshape(B * T, 3, 64, 64)
    emb = M["enc"](flat)[0].view(B, T, -1)
    state = M["rssm"].initial_state(B, device=device)
    for t in range(T):
        state, info = M["rssm"].observe(state, acts[:, t], emb[:, t])
        k = kl_gaussian(info["prior_mu"], info["prior_logvar"],
                        info["post_mu"], info["post_logvar"], M["rssm"].min_std)
        kls.append(k.mean(dim=-1).cpu().numpy())  # 每样本每步的 KL（每元素均值）
kls = np.concatenate(kls)
print(f"[DIAG] 检查点: {ckpt_path}")
print(f"[DIAG] 真实 KL（每元素均值）: mean={kls.mean():.4f} median={np.median(kls):.4f} "
      f"p10={np.percentile(kls,10):.4f} p90={np.percentile(kls,90):.4f} max={kls.max():.4f}")
for fb in (0.5, 0.2, 0.1, 0.05, 0.01):
    active = np.mean(kls > fb) * 100
    print(f"[DIAG]   相对 free_bits={fb}: 有梯度样本占比 {active:.1f}%")
