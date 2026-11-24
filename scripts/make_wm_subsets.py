#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为世界模型样本效率实验切分离线数据子集（按轨迹数 10/30/50/100%）。

源: outputs/data/trajectories.h5 (100 条轨迹)。
固定 seed 按质量分布均匀抽样，保证各档为全量数据的严格子集（嵌套）。
输出: outputs/data/subsets/trajs_{pct}.h5，结构与源完全一致。
"""
import os
import sys
import json
import h5py
import numpy as np

ROOT = "/data/wzh/robot/vla-world-model-lab"
SRC = os.path.join(ROOT, "outputs/data/trajectories.h5")
OUT_DIR = os.path.join(ROOT, "outputs/data/subsets")
PCTS = [10, 30, 50, 100]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(SRC, "r") as f:
        keys = sorted(f.keys())
        attrs = dict(f.attrs)
    n = len(keys)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)  # 固定排列 -> 各比例为嵌套子集
    print(f"[SRC] n_traj={n} attrs={ {k: (v.tolist() if hasattr(v,'tolist') else v) for k,v in attrs.items()} }")

    for pct in PCTS:
        k = max(1, round(n * pct / 100.0))
        sel = sorted([keys[i] for i in perm[:k]])  # 按原名排序保证确定性
        dst = os.path.join(OUT_DIR, f"trajs_{pct}.h5")
        n_frames = 0
        with h5py.File(SRC, "r") as fsrc, h5py.File(dst, "w") as fdst:
            for key in sel:
                gsrc = fsrc[key]
                gdst = fdst.create_group(key)
                for ds in gsrc.keys():
                    gdst.create_dataset(ds, data=gsrc[ds][:])
                n_frames += gsrc["actions"].shape[0]
            for a, v in attrs.items():
                fdst.attrs[a] = v
        print(f"[SUB {pct:3d}%] traj={k:3d} frames={n_frames:4d} -> {dst}", flush=True)
        with open(dst.replace(".h5", ".json"), "w") as jf:
            json.dump({"pct": pct, "n_traj": k, "n_frames": n_frames,
                       "keys": sel}, jf, indent=2)
    print("[DONE]")


if __name__ == "__main__":
    main()
