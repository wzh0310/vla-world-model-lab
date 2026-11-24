#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""标准化数据格式化：把混合采集数据统一为标准轨迹格式并按 9:1 划分训练/测试集。

运行容器环境：vla（仅需 h5py/numpy，无需 mujoco）。

标准格式（每条轨迹组 traj_XXXX）：
  observations     : (T,224,224,3) uint8   观测帧 x_t
  images           : (T,224,224,3) uint8   与 observations 同值（兼容既有采样器）
  actions          : (T,4) float32         动作 a_t
  rewards          : (T,) float32          奖励 r_t
  next_observations: (T,224,224,3) uint8   下一帧 x_{t+1}（末帧自复制）
  dones            : (T,) bool             终止标记
  states           : (T,13) float32        低维状态（奖励口径核对用）

即统一的 (observation, action, reward, next_observation, done) 转移元组。
划分按"轨迹"粒度随机打乱后 9:1 切分，避免同一轨迹跨训练/测试造成泄漏。

用法：
  python scripts/format_dataset.py \
      --raw outputs/data/mixed_raw.h5 \
      --extra outputs/data/trajectories.h5 \
      --out-dir outputs/data/wm_dataset
"""
import argparse
import os

import h5py
import numpy as np


def read_episodes(path):
    """读取一个 HDF5 的全部轨迹组，返回 (episodes, root_attrs)。"""
    eps = []
    with h5py.File(path, "r") as f:
        attrs = {k: np.asarray(v) for k, v in f.attrs.items()}
        for k in sorted(f.keys()):
            g = f[k]
            ep = dict(
                images=g["images"][:],
                actions=g["actions"][:].astype(np.float32),
                rewards=g["rewards"][:].astype(np.float32),
                dones=g["dones"][:].astype(bool),
                source=str(g.attrs.get("source", "unknown")),
            )
            if "states" in g:
                ep["states"] = g["states"][:].astype(np.float32)
            else:
                ep["states"] = np.zeros((len(ep["images"]), 13), np.float32)
            eps.append(ep)
    return eps, attrs


def write_split(path, episodes, attrs, name):
    with h5py.File(path, "w") as f:
        f.attrs["split"] = name
        f.attrs["num_episodes"] = len(episodes)
        f.attrs["total_frames"] = int(sum(len(e["images"]) for e in episodes))
        f.attrs["fps"] = 50
        f.attrs["image_shape"] = np.asarray([224, 224, 3])
        f.attrs["action_low"] = np.asarray(attrs.get("action_low", [-0.02, -0.02, -0.02, 0.0]), np.float32)
        f.attrs["action_high"] = np.asarray(attrs.get("action_high", [0.02, 0.02, 0.02, 1.0]), np.float32)
        sources = {}
        for i, ep in enumerate(episodes):
            g = f.create_group(f"traj_{i:04d}")
            g.attrs["source"] = ep["source"]
            T = len(ep["images"])
            # next_observations = x_{t+1}，末帧自复制（done 帧的 next 无意义，保持形状一致）
            nxt = np.empty_like(ep["images"])
            nxt[:-1] = ep["images"][1:]
            nxt[-1] = ep["images"][-1]
            g.create_dataset("observations", data=ep["images"], dtype="u1", compression="lzf")
            g.create_dataset("images", data=ep["images"], dtype="u1", compression="lzf")
            g.create_dataset("next_observations", data=nxt, dtype="u1", compression="lzf")
            g.create_dataset("actions", data=ep["actions"], dtype="f4")
            g.create_dataset("rewards", data=ep["rewards"], dtype="f4")
            g.create_dataset("dones", data=ep["dones"], dtype=bool)
            g.create_dataset("states", data=ep["states"], dtype="f4")
            sources[ep["source"]] = sources.get(ep["source"], 0) + 1
        f.attrs["sources"] = ",".join(f"{k}:{v}" for k, v in sorted(sources.items()))
    print(f"[FMT] {name}: {len(episodes)} 条轨迹, "
          f"{sum(len(e['images']) for e in episodes)} 帧 -> {path} "
          f"({os.path.getsize(path)/1e6:.1f} MB) sources={sources}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="世界模型数据集标准化与 9:1 划分")
    ap.add_argument("--raw", default="/data/wzh/robot/vla-world-model-lab/"
                                     "outputs/data/mixed_raw.h5")
    ap.add_argument("--extra", default="/data/wzh/robot/vla-world-model-lab/"
                                       "outputs/data/trajectories.h5",
                    help="可选：合并既有专家/随机轨迹以扩充数据；传 none 跳过")
    ap.add_argument("--out-dir", default="/data/wzh/robot/vla-world-model-lab/"
                                         "outputs/data/wm_dataset")
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_eps, attrs = read_episodes(args.raw)
    print(f"[FMT] 混合采集: {len(all_eps)} 条轨迹, "
          f"{sum(len(e['images']) for e in all_eps)} 帧", flush=True)
    if args.extra and args.extra.lower() != "none" and os.path.isfile(args.extra):
        extra, extra_attrs = read_episodes(args.extra)
        all_eps += extra
        print(f"[FMT] 合并既有数据: +{len(extra)} 条轨迹 "
              f"({sum(len(e['images']) for e in extra)} 帧)", flush=True)
        # 动作空间口径以混合采集数据为准（两者一致）

    n = len(all_eps)
    idx = np.random.default_rng(args.seed).permutation(n)
    n_test = max(1, int(round(n * args.test_ratio)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train_eps = [all_eps[i] for i in train_idx]
    test_eps = [all_eps[i] for i in test_idx]

    write_split(os.path.join(args.out_dir, "train.h5"), train_eps, attrs, "train")
    write_split(os.path.join(args.out_dir, "test.h5"), test_eps, attrs, "test")

    total = sum(len(e["images"]) for e in all_eps)
    print(f"[DONE] 共 {n} 条 / {total} 帧, 划分 {len(train_eps)}:{len(test_eps)} "
          f"(train:test≈{1-args.test_ratio:.1f}:{args.test_ratio:.1f})", flush=True)


if __name__ == "__main__":
    main()
