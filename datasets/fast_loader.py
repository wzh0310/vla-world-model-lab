#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GPU 内存预加载序列采样器。

问题背景：EpisodicDataset.__getitem__ 对每个样本都在 CPU 上逐帧 resize
20 张 224x224 图像，num_workers=0 时串行执行，GPU 大部分时间在等数据
（实测利用率 ~1%，100 步耗时 400+ 秒）。

方案：训练开始前一次性把所有轨迹图像 resize 到 64x64 并常驻显存
（100 条约 82MB），训练循环直接从显存切片组 batch，纯 GPU 运算。
实测 1000 步从 ~70 分钟降到 52 秒。

注意：采样器按「序列起点索引」随机取样，语义与 EpisodicDataset 一致
（单轨迹内滑窗、不跨回合、动作已归一化），不依赖 torch DataLoader。
"""
import numpy as np
import h5py
import torch
import torch.nn.functional as F


class GPUSequenceSampler:
    """常驻显存的序列采样器，接口对齐 DataLoader 的迭代语义。"""

    def __init__(self, hdf5_path, seq_len=20, image_size=64, device="cuda", seed=0):
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.device = device
        self.rng = np.random.default_rng(seed)
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
        out = (torch.stack(imgs, 0), torch.stack(acts, 0))
        if with_rewards:
            out = out + (torch.stack(rews, 0),)
        return out
