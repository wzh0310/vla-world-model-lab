#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基础验证：随机初始化的隐空间策略在世界模型想象 rollout 中可正常输出动作。

验证内容（使用 vla 环境）：
  1) 从检查点加载世界模型，读取其真实隐状态维度 feat_dim；
  2) 用真实轨迹的前 K 帧 observe 建立后验起点（teacher-forcing，仅初始化）；
  3) 随机初始化 ActorCritic，在 H 步纯想象 rollout 中逐步：
       feat -> actor.sample() -> 归一化动作 [-1,1] -> imagine_step -> 新隐状态
     全程不接入任何真实图像；
  4) 逐项断言：动作维度、范围、价值标量维度、ActionScaler 映射回真实范围正确；
  5) 附带检查 reparameterization 梯度可沿想象链回传到策略参数。

用法（vla 环境）：
  python trainer/verify_actor_critic.py \
      --ckpt outputs/abl2_stoch48/checkpoints/wm_best.pt \
      --data outputs/data/trajectories.h5
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.world_model import WorldModel
from models.actor_critic import build_from_world_model, ActionScaler


def parse_args():
    ap = argparse.ArgumentParser(description="隐空间策略想象 rollout 基础验证")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/abl2_stoch48/checkpoints/wm_best.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--traj", default=None, help="指定轨迹 key；默认取数据集第一个")
    ap.add_argument("--warmup", type=int, default=3, help="后验热身帧数 K")
    ap.add_argument("--horizon", type=int, default=10, help="想象步数 H")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_trajectory(hdf5_path, device, key=None):
    """读取一条轨迹：归一化图像 [-1,1] (T,3,64,64) + 归一化动作 (T,4)。"""
    with h5py.File(hdf5_path, "r") as f:
        if key is None:
            key = next(k for k in f.keys() if isinstance(f[k], h5py.Group))
        low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
        high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
        span = torch.where(high == low, torch.ones_like(high), high - low)
        raw = torch.from_numpy(f[key]["images"][:]).permute(0, 3, 1, 2).float().to(device)
        small = F.interpolate(raw, size=(64, 64), mode="bilinear", align_corners=False)
        imgs = small / 127.5 - 1.0
        acts = torch.as_tensor(f[key]["actions"][:], dtype=torch.float32, device=device)
        acts = ((acts - low) / span * 2.0 - 1.0).clamp(-1.0, 1.0)
    return key, imgs, acts, low, high


@torch.no_grad()
def posterior_start(wm, imgs, acts, K):
    """用前 K 帧真实图像 + 动作 doing observe，得到想象起点的 RSSM 状态。"""
    state = wm.rssm.initial_state(1, device=wm.device)
    for t in range(K):
        embed = wm._encode(imgs[t].unsqueeze(0))         # (1, latent)
        state, _ = wm.rssm.observe(state, acts[t].unsqueeze(0), embed)
    return state


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    # 1) 加载世界模型 + 读取真实隐维度
    wm = WorldModel.from_checkpoint(args.ckpt, device=str(dev))
    Fdim = wm.feat_dim                                   # 严格对齐 RSSM

    # 2) 策略网络：随机初始化，维度来自 wm
    ac = build_from_world_model(wm).to(dev)
    scaler = ActionScaler.from_arrays(
        [-0.02, -0.02, -0.02, 0.0], [0.02, 0.02, 0.02, 1.0], device=dev)

    # 3) 真实轨迹建立后验起点
    key, imgs, acts, low, high = load_trajectory(args.data, dev, args.traj)
    K = args.warmup
    if imgs.shape[0] < K + 1:
        raise ValueError(f"轨迹 {key} 长度 {imgs.shape[0]} 不足以热身 {K} 帧")
    state = posterior_start(wm, imgs, acts, K)
    print(f"[VERIFY] 轨迹={key} 隐维度={Fdim} 热身={K}帧 想象={args.horizon}步 设备={dev}", flush=True)

    # 4) 纯想象 rollout：逐步输出动作（不接真实图像）
    act_list, val_list = [], []
    for t in range(args.horizon):
        feat = wm.rssm.get_feature(state)                 # (1, Fdim)
        mu, log_std = ac.actor(feat)                      # 分布参数
        act = ac.actor.sample(feat)                       # 采样动作 [-1,1]
        val = ac.values(feat)                             # 状态价值
        assert act.shape == (1, 4), act.shape
        assert val.shape == (1,), val.shape
        real = scaler.to_env(act)                         # 映射回真实范围
        assert real.shape == (1, 4)
        assert real[:, :3].abs().max().item() <= 0.02 + 1e-6, "位置动作超出 [-0.02,0.02]"
        assert real[:, 3].min().item() >= -1e-6 and real[:, 3].max().item() <= 1.0 + 1e-6
        act_list.append(act.detach())
        val_list.append(val.detach())
        state, _ = wm.rssm.imagine_step(state, act, sample=True)

    acts_out = torch.cat(act_list, 0)                     # (H,4)
    vals_out = torch.cat(val_list, 0)                     # (H,)
    assert acts_out.shape == (args.horizon, 4)
    assert acts_out.abs().max().item() <= 1.0 + 1e-6

    # 5) 梯度链路检查：奖励头经想象链回传到策略参数
    ac.train()
    for p in wm.rssm.parameters():
        p.requires_grad_(False)
    if wm.reward_head is not None:
        for p in wm.reward_head.parameters():
            p.requires_grad_(False)
    s = posterior_start(wm, imgs, acts, K)
    total = 0.0
    for t in range(args.horizon):
        feat = wm.rssm.get_feature(s)
        act = ac.actor.sample(feat)
        s, _ = wm.rssm.imagine_step(s, act, sample=True)
        if wm.reward_head is not None:
            total = total + wm.reward_head(wm.rssm.get_feature(s)).squeeze()
    if isinstance(total, torch.Tensor):
        total.backward()
        grads = [p.grad for p in ac.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum().item() > 0 for g in grads), "策略参数无梯度"
        print(f"[VERIFY] 梯度链路 OK：{len(grads)}/{sum(1 for _ in ac.parameters())} 个参数收到非零梯度", flush=True)

    print(f"[VERIFY] 想象 rollout OK：{args.horizon} 步动作 shape={tuple(acts_out.shape)} "
          f"范围=[{acts_out.min():.3f},{acts_out.max():.3f}] 价值 shape={tuple(vals_out.shape)} "
          f"策略参数量={ac.count_parameters():,}", flush=True)
    print(f"[VERIFY] 前3步归一化动作:\n{acts_out[:3].cpu().numpy().round(3)}", flush=True)
    print("ACTOR_CRITIC_IMAGINATION_OK", flush=True)


if __name__ == "__main__":
    main()
