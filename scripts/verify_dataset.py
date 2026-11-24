#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据质量自检：统计规模、来源分布、初始位置/动作幅度多样性、9:1 划分正确性。

运行容器环境：vla。
"""
import os
import sys

import h5py
import numpy as np


def stats(path):
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        n_traj = len(keys)
        sources, frames, starts, amps = {}, [], [], []
        for k in keys:
            g = f[k]
            src = str(g.attrs.get("source", "unknown"))
            sources[src] = sources.get(src, 0) + 1
            imgs, acts = g["images"], g["actions"]
            frames.append(imgs.shape[0])
            # states 前 7 维是关节角，7:9 是 cube xy，若存在则记录初始位置
            if "states" in g and g["states"].shape[-1] >= 9:
                starts.append(g["states"][0, 7:9])
            amps.append(np.abs(acts[:, :3]).max())
        print(f"[{os.path.basename(path)}] 轨迹={n_traj} 总帧={sum(frames)} "
              f"平均长度={np.mean(frames):.1f} 最短={min(frames)} 最长={max(frames)}")
        print(f"  来源分布: {sources}")
        if starts:
            s = np.array(starts)
            print(f"  立方体初始位置范围: x[{s[:,0].min():.3f},{s[:,0].max():.3f}] "
                  f"y[{s[:,1].min():.3f},{s[:,1].max():.3f}] "
                  f"(x跨度={np.ptp(s[:,0]):.3f}, y跨度={np.ptp(s[:,1]):.3f})")
        a = np.array(amps)
        print(f"  动作峰值幅度: mean={a.mean():.4f} "
              f"range=[{a.min():.4f},{a.max():.4f}] (幅度多样性)")
        return n_traj, sum(frames)


def verify_split_keys(train_path, test_path):
    def keys(p):
        with h5py.File(p, "r") as f:
            # 对比轨迹状态序列的哈希，确保 train/test 无重叠轨迹
            return [hash(f[k]["states"][:].tobytes()) if "states" in f[k]
                    else hash(f[k]["images"][:].tobytes()) for k in f.keys()]
    tr, te = set(keys(train_path)), set(keys(test_path))
    overlap = tr & te
    print(f"[划分校验] train={len(tr)} test={len(te)} 重叠轨迹={len(overlap)} "
          f"{'OK 无泄漏' if not overlap else 'ERROR 存在数据泄漏'}")
    return len(overlap) == 0


def main():
    base = "/data/wzh/robot/vla-world-model-lab/outputs/data"
    ds = os.path.join(base, "wm_dataset")
    raw = os.path.join(base, "mixed_raw.h5")
    if os.path.isfile(raw):
        print("=== 原始混合采集数据 ===")
        stats(raw)
    print("=== 格式化后 9:1 划分 ===")
    stats(os.path.join(ds, "train.h5"))
    stats(os.path.join(ds, "test.h5"))
    ok = verify_split_keys(os.path.join(ds, "train.h5"), os.path.join(ds, "test.h5"))
    print("VERIFY", "OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
