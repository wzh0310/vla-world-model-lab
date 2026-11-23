#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出世界模型离线轨迹数据（HDF5 格式）。

数据来源：
1. 专家轨迹：mujoco-studio-lab/data/raw_demos_v2.npz（40 条精选成功轨迹）
2. 随机策略轨迹：PushCubeEnv 随机采样（补足至 100 条，覆盖状态空间）

用法（mujoco conda 环境）：
  /opt/miniconda3/envs/mujoco/bin/python scripts/export_trajectories.py \
      --expert /data/wzh/robot/mujoco-studio-lab/data/raw_demos_v2.npz \
      --out outputs/data/trajectories.h5 --random-episodes 60 --random-steps 30
"""
import argparse
import os
import sys

import h5py
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDIO = "/data/wzh/robot/mujoco-studio-lab"

# 与 PushCubeEnv 一致的奖励常量（近似：专家轨迹无逐帧 EE-cube 距离，省略该项）
K_DIST, K_EE, TIME_PENALTY = 10.0, 0.1, 0.01
SUCCESS_RADIUS, SUCCESS_BONUS, FAIL_PENALTY = 0.03, 10.0, -1.0
ACTION_LOW = np.array([-0.02, -0.02, -0.02, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([0.02, 0.02, 0.02, 1.0], dtype=np.float32)


def rewards_from_states(states):
    """从状态序列近似重建逐帧奖励与终止标记。"""
    cube, target = states[:, 7:9], states[:, 10:12]
    dist = np.linalg.norm(cube - target, axis=1)
    prev = np.concatenate([[dist[0]], dist[:-1]])
    r_shaping = -K_DIST * (dist - prev)
    r_time = -TIME_PENALTY
    success = dist < SUCCESS_RADIUS
    fell = states[:, 9] < 0.70  # cube z 低于桌面
    rewards = (r_shaping + r_time + success * SUCCESS_BONUS + fell * FAIL_PENALTY).astype(np.float32)
    dones = (success | fell).astype(bool)
    dones[-1] = True
    return rewards, dones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default=os.path.join(STUDIO, "data/raw_demos_v2.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--random-episodes", type=int, default=60)
    ap.add_argument("--random-steps", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    raw = np.load(args.expert)
    images, actions, states, lengths = (raw["images"], raw["actions"], raw["states"], raw["episode_lengths"])
    print(f"[expert] {len(lengths)} 条轨迹 / {len(images)} 帧, 图像 {images.shape[1:]}", flush=True)

    episodes, cursor = [], 0
    for length in lengths:
        length = int(length)
        ep = dict(images=images[cursor:cursor + length], actions=actions[cursor:cursor + length],
                  states=states[cursor:cursor + length], source="expert")
        ep["rewards"], ep["dones"] = rewards_from_states(ep["states"])
        episodes.append(ep)
        cursor += length

    # 随机策略采样补足数据量（无需专家质量，覆盖状态空间即可）
    if args.random_episodes > 0:
        from pushcube_envs.push_cube_env import PushCubeEnv
        env = PushCubeEnv(obs_mode="rgb", rgb_preset="demo", max_episode_steps=args.random_steps,
                          randomize_start=True)
        rng = np.random.default_rng(42)
        for ep_i in range(args.random_episodes):
            obs, _ = env.reset(seed=5000 + ep_i)
            ims, acts, sts, rws, dns = [], [], [], [], []
            for step in range(args.random_steps):
                action = np.clip(rng.uniform(-0.02, 0.02, size=3).astype(np.float32), ACTION_LOW[:3], ACTION_HIGH[:3])
                action = np.concatenate([action, [float(rng.integers(0, 2))]]).astype(np.float32)
                img = env.get_rgb_image(preset="demo", chw=False, normalize=False)
                ims.append(np.asarray(img, dtype=np.uint8))
                acts.append(action)
                sts.append(env._get_state())
                obs, reward, term, trunc, info = env.step(action)
                rws.append(float(reward))
                dns.append(bool(term or trunc))
                if term or trunc:
                    break
            episodes.append(dict(images=np.stack(ims), actions=np.stack(acts), states=np.stack(sts),
                                 rewards=np.asarray(rws, np.float32), dones=np.asarray(dns, bool),
                                 source="random"))
            if (ep_i + 1) % 10 == 0:
                print(f"[random] {ep_i + 1}/{args.random_episodes} 条", flush=True)
        env.close()

    with h5py.File(args.out, "w") as f:
        f.attrs["num_episodes"] = len(episodes)
        f.attrs["fps"] = 50
        f.attrs["image_shape"] = images.shape[1:]
        f.attrs["action_low"] = ACTION_LOW
        f.attrs["action_high"] = ACTION_HIGH
        f.attrs["sources"] = "expert+random"
        for i, ep in enumerate(episodes):
            g = f.create_group(f"traj_{i:04d}")
            g.attrs["source"] = ep["source"]
            g.create_dataset("images", data=ep["images"], dtype="u1", compression="lzf")
            g.create_dataset("actions", data=ep["actions"], dtype="f4")
            g.create_dataset("states", data=ep["states"], dtype="f4")
            g.create_dataset("rewards", data=ep["rewards"], dtype="f4")
            g.create_dataset("dones", data=ep["dones"], dtype=bool)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[done] {len(episodes)} 条轨迹 -> {args.out} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
