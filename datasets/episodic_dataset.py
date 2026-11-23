#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回合序列数据集：HDF5 轨迹 -> (image_seq, action_seq, reward_seq, done_seq)。

- 每条序列固定 20 步（可配置），只从单条轨迹内滑动截取，不跨回合。
- 图像 224×224 -> 64×64 双线性下采样，归一化到 [-1, 1]。
- 动作按环境动作空间 [lo, hi] 归一化到 [-1, 1]，与策略训练一致。
- 奖励缩放 1/10（稀疏 +10 成功奖励 -> +1），done 转 float32 供损失掩码使用。
"""
import os
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodicDataset(Dataset):
    """从 HDF5 离线轨迹中采样固定长度序列样本。"""

    def __init__(self, hdf5_path: str, seq_len: int = 20, image_size: int = 64,
                 stride: int = 1, source: Optional[str] = None,
                 action_low: Optional[np.ndarray] = None,
                 action_high: Optional[np.ndarray] = None):
        super().__init__()
        if not os.path.isfile(hdf5_path):
            raise FileNotFoundError(f"轨迹文件不存在: {hdf5_path}")
        if seq_len < 2 or image_size < 16:
            raise ValueError("seq_len 至少 2，image_size 至少 16")
        self.hdf5_path = hdf5_path
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.stride = int(stride)
        self.source = source

        self._file = None  # 延迟打开，避免 DataLoader worker fork 时继承句柄
        with h5py.File(hdf5_path, "r") as f:
            self.action_low = np.asarray(action_low if action_low is not None else f.attrs["action_low"], np.float32)
            self.action_high = np.asarray(action_high if action_high is not None else f.attrs["action_high"], np.float32)
            # 预构建索引 (轨迹键, 起始帧)，只统计长度足够的轨迹
            self._index = []
            for key in sorted(f.keys()):
                if source is not None and f[key].attrs.get("source") != source:
                    continue
                length = f[key]["images"].shape[0]
                for start in range(0, length - self.seq_len + 1, self.stride):
                    self._index.append((key, start))

        if not self._index:
            raise RuntimeError(f"没有可用序列样本（seq_len={self.seq_len}, source={source}）")
        self._stats = {"num_sequences": len(self._index),
                       "seq_len": self.seq_len, "image_size": self.image_size}

    def __len__(self) -> int:
        return len(self._index)

    def _ensure_file(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

    @staticmethod
    def _resize_images(images: np.ndarray, size: int) -> np.ndarray:
        """(T,H,W,3) uint8 -> (T,3,size,size) float32 ∈ [-1,1]，使用 torchvision 避免额外依赖 cv2。"""
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float()  # (T,3,H,W)
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return (x / 127.5 - 1.0).numpy().astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        self._ensure_file()
        key, start = self._index[idx]
        g = self._file[key]
        end = start + self.seq_len
        images = g["images"][start:end]
        actions = g["actions"][start:end].astype(np.float32)
        rewards = g["rewards"][start:end].astype(np.float32)
        dones = g["dones"][start:end].astype(np.float32)

        images = self._resize_images(images, self.image_size)
        span = self.action_high - self.action_low
        actions = (actions - self.action_low) / np.where(span == 0, 1.0, span)
        actions = np.clip(actions * 2.0 - 1.0, -1.0, 1.0)
        rewards = np.clip(rewards / 10.0, -1.0, 1.0)

        return {"image_seq": images, "action_seq": actions,
                "reward_seq": rewards[..., None], "done_seq": dones[..., None]}

    def __del__(self):
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            except Exception:
                pass


def get_dataloader(hdf5_path: str, batch_size: int = 32, num_workers: int = 0,
                   shuffle: bool = True, **kwargs):
    """便捷工厂：返回 PyTorch DataLoader。"""
    from torch.utils.data import DataLoader
    ds = EpisodicDataset(hdf5_path, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..",
                                                   "outputs/data/trajectories.h5"))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    loader = get_dataloader(args.data, batch_size=args.batch_size)
    batch = next(iter(loader))
    print("序列数:", len(loader.dataset), "批次大小:", args.batch_size)
    for name, tensor in batch.items():
        print(f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
              f"min={float(tensor.min()):.3f} max={float(tensor.max()):.3f}")
    assert batch["image_seq"].shape[2:] == (3, 64, 64)
    assert batch["image_seq"].min() >= -1.01 and batch["image_seq"].max() <= 1.01
    assert batch["action_seq"].min() >= -1.01 and batch["action_seq"].max() <= 1.01
    print("SELFTEST_OK")
