#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GPU 内存预加载序列采样器（支持可选图像增强）。

在原有显存预加载基础上新增图像增强开关（--augment），用于提升世界模型
多步开环预测的泛化性。增强设计遵循两条原则：
  1. 整段序列共享同一组增强参数（平移/亮度）——避免逐帧随机扰动破坏 RSSM
     依赖的时序连贯性；
  2. 平移幅度小（默认 ±2 像素 / 64×64）、亮度扰动温和（默认 ±0.1）——
     保住推立方体任务中立方体的绝对位置信息，避免增强引入标签噪声。

增强只在训练 sample() 时施加，评估脚本用原始（未增强）数据，保证指标可比。
默认 augment=False，与既有 AE 训练脚本接口完全兼容。
"""
import numpy as np
import h5py
import torch
import torch.nn.functional as F


class GPUSequenceSampler:
    """常驻显存的序列采样器，接口对齐 DataLoader 的迭代语义。"""

    def __init__(self, hdf5_path, seq_len=20, image_size=64, device="cuda", seed=0,
                 augment=False, shift_px=2, brightness=0.1):
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.augment = bool(augment)
        self.shift_px = float(shift_px)
        self.brightness = float(brightness)
        if self.augment:
            print(f"[AUG] 图像增强开启: 平移±{self.shift_px}px 亮度±{self.brightness} "
                  f"(整段序列共享参数)", flush=True)
        t_load = __import__("time").time()

        with h5py.File(hdf5_path, "r") as f:
            low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
            high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
            span = torch.where(high == low, torch.ones_like(high), high - low)
            keys = sorted(f.keys())
            raw = np.concatenate([f[k]["images"][:] for k in keys], axis=0)  # (N,224,224,3) uint8
            raw_t = torch.from_numpy(raw).permute(0, 3, 1, 2).float().to(device)
            del raw
            small = F.interpolate(raw_t, size=(image_size, image_size),
                                  mode="bilinear", align_corners=False)
            small = (small / 127.5 - 1.0).half()  # (N,3,S,S) float16 in [-1,1]
            del raw_t
            self.images, self.actions, self.rewards = [], [], []
            index = []
            offset = 0
            for tid, k in enumerate(keys):
                n = f[k]["images"].shape[0]
                self.images.append(small[offset:offset + n])
                a = torch.as_tensor(f[k]["actions"][:], dtype=torch.float32, device=device)
                self.actions.append(((a - low) / span * 2 - 1).clamp(-1, 1))
                r = torch.as_tensor(f[k]["rewards"][:], dtype=torch.float32, device=device)
                self.rewards.append((r / 10.0).clamp(-1, 1)[:, None])  # (n,1)
                for s in range(0, n - seq_len + 1):
                    index.append((tid, s))
                offset += n

        self.index = np.asarray(index, dtype=np.int64)
        self.mem_mb = small.element_size() * small.numel() / 1e6
        self.load_sec = __import__("time").time() - t_load

    def _augment_batch(self, imgs):
        """对 batch 施加增强：(B,T,3,S,S) -> 同形状，整段共享参数。"""
        B, T, C, H, W = imgs.shape
        dev = imgs.device
        # 每个序列一组平移（归一化到 [-1,1] 的 grid 位移）
        tx = torch.empty(B, device=dev).uniform_(-self.shift_px, self.shift_px) / W * 2
        ty = torch.empty(B, device=dev).uniform_(-self.shift_px, self.shift_px) / H * 2
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        theta_bt = theta.repeat_interleave(T, dim=0)              # (B*T,2,3)
        x = imgs.reshape(B * T, C, H, W)
        grid = F.affine_grid(theta_bt, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        x = x.reshape(B, T, C, H, W)
        # 亮度/对比度扰动：整段共享，形状 (B,1,1,1,1) 广播到全序列
        scale = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(1 - self.brightness,
                                                                 1 + self.brightness)
        bias = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(-self.brightness / 2,
                                                                self.brightness / 2)
        return (x * scale + bias).clamp(-1.0, 1.0)

    def __len__(self):
        return len(self.index)

    def sample(self, batch_size, with_rewards=False):
        """随机取一个 batch（有放回），返回图像 (B,T,3,S,S) + 动作 (B,T,4)。

        with_rewards=True 时额外返回归一化奖励 (B,T,1)，供世界模型奖励头训练。
        默认 False 保持与 AE 训练脚本的两元组接口兼容。
        """
        sel = self.index[self.rng.integers(0, len(self.index), size=batch_size)]
        imgs, acts, rews = [], [], []
        for tid, s in sel:
            t, s = int(tid), int(s)
            imgs.append(self.images[t][s:s + self.seq_len].float())
            acts.append(self.actions[t][s:s + self.seq_len])
            if with_rewards:
                rews.append(self.rewards[t][s:s + self.seq_len])
        img_batch = torch.stack(imgs, 0)
        if self.augment:
            img_batch = self._augment_batch(img_batch)
        out = (img_batch, torch.stack(acts, 0))
        if with_rewards:
            out = out + (torch.stack(rews, 0),)
        return out
