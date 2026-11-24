#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""混合策略数据采集：随机策略 + 已训练 PPO 策略混合采集 100 条交互轨迹。

运行容器环境：mujoco（含 mujoco4 + sb3 + torch，可加载 PPO 与离屏渲染）。

多样性设计：
  - 立方体初始位置：randomize_start=True 且 start_xy_range 调大（覆盖更大初始分布）
  - 动作幅度：随机策略按 episode 采样幅度缩放（0.4~1.0），PPO 策略叠加小噪声
  - 两种来源各占一半，兼顾"状态空间覆盖"与"任务相关轨迹"

输出原始 HDF5（后续由 format_dataset.py 在 vla 环境统一为标准格式并 9:1 划分）。
"""
import argparse
import os
import sys

import h5py
import numpy as np

ZOO = "/data/wzh/robot/embodied-policy-zoo"
sys.path.insert(0, ZOO)
from envs.paths import setup_sys_path  # noqa: E402

setup_sys_path()
import gymnasium as gym  # noqa: E402
import pushcube_envs  # noqa: E402,F401
from stable_baselines3 import PPO  # noqa: E402

DEFAULT_PPO = ("/data/wzh/robot/outputs/ppo_300k/"
               "ppo_state_300k_s0/checkpoints/ppo_state_100000_steps.zip")


def scale_action(action, scale, rng, low, high):
    """对动作前 3 维（平移）按 episode 幅度缩放，gripper 保持不变，再裁剪到动作空间。"""
    a = np.asarray(action, dtype=np.float32).copy()
    a[:3] = np.clip(a[:3] * scale, low[:3], high[:3])
    a[3] = float(np.clip(a[3], low[3], high[3]))
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="混合策略采集世界模型交互数据")
    ap.add_argument("--out", default="/data/wzh/robot/vla-world-model-lab/"
                                     "outputs/data/mixed_raw.h5")
    ap.add_argument("--n-ppo", type=int, default=50, help="PPO 策略轨迹数")
    ap.add_argument("--n-random", type=int, default=50, help="随机策略轨迹数")
    ap.add_argument("--max-steps", type=int, default=50, help="每条轨迹最大步数")
    ap.add_argument("--ppo-ckpt", default=DEFAULT_PPO)
    ap.add_argument("--start-range", type=float, default=0.12,
                    help="立方体初始位置随机范围（米），默认比训练时的 0.03 更大")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    env = gym.make("PushCube-v0", obs_mode="both", disable_env_checker=True,
                   randomize_start=True, start_xy_range=args.start_range,
                   rgb_preset="demo", max_episode_steps=args.max_steps)
    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)

    ppo = PPO.load(args.ppo_ckpt, device="cpu")
    print(f"[COLLECT] PPO 已加载: {args.ppo_ckpt}", flush=True)
    print(f"[COLLECT] 采集计划: PPO {args.n_ppo} + 随机 {args.n_random} 条, "
          f"每条≤{args.max_steps}步, start_range={args.start_range}", flush=True)

    episodes = []
    total_frames = 0
    plan = (["ppo"] * args.n_ppo) + (["random"] * args.n_random)
    rng.shuffle(plan)  # 打乱来源顺序，避免数据分布按来源聚集

    for ep_i, source in enumerate(plan):
        obs, info = env.reset(seed=args.seed * 1000 + ep_i)
        # 每个 episode 采样一个动作幅度缩放，保证"不同动作幅度"的多样性
        amp = float(rng.uniform(0.4, 1.0)) if source == "random" else 1.0
        noise = float(rng.uniform(0.0, 0.15)) if source == "ppo" else 0.0
        ims, acts, sts, rws, dns = [], [], [], [], []
        for t in range(args.max_steps):
            img = np.asarray(obs["image"], dtype=np.uint8)  # (224,224,3)
            state = np.asarray(obs["state"], dtype=np.float32)
            if source == "ppo":
                a, _ = ppo.predict(state, deterministic=True)
                a = np.asarray(a, dtype=np.float32)
                if noise > 0:
                    a = a + rng.normal(0, 0.02 * noise, size=4).astype(np.float32)
                action = scale_action(a, 1.0, rng, low, high)
            else:
                # 随机策略：带轻微朝目标(+x)偏置，避免完全原地打转；幅度由 amp 控制
                a = rng.uniform(-0.02, 0.02, size=3).astype(np.float32)
                a[0] += 0.008  # 朝目标 +x 的弱偏置
                a = np.concatenate([a, [float(rng.integers(0, 2))]]).astype(np.float32)
                action = scale_action(a, amp, rng, low, high)
            ims.append(img)
            acts.append(action)
            sts.append(state)
            obs, reward, term, trunc, info = env.step(action)
            rws.append(float(reward))
            done = bool(term or trunc)
            dns.append(done)
            if done:
                break
        ep = dict(images=np.stack(ims), actions=np.stack(acts), states=np.stack(sts),
                  rewards=np.asarray(rws, np.float32), dones=np.asarray(dns, bool),
                  source=source)
        episodes.append(ep)
        total_frames += len(ims)
        if (ep_i + 1) % 10 == 0:
            succ = bool(np.any(ep["dones"]) and info.get("success", False))
            print(f"[COLLECT] {ep_i + 1}/{len(plan)} 条 "
                  f"({source}, {len(ims)}帧, amp={amp:.2f}, succ={succ})", flush=True)

    env.close()

    with h5py.File(args.out, "w") as f:
        f.attrs["num_episodes"] = len(episodes)
        f.attrs["total_frames"] = total_frames
        f.attrs["fps"] = 50
        f.attrs["image_shape"] = [224, 224, 3]
        f.attrs["action_low"] = low
        f.attrs["action_high"] = high
        f.attrs["sources"] = "ppo+random"
        f.attrs["seed"] = args.seed
        for i, ep in enumerate(episodes):
            g = f.create_group(f"traj_{i:04d}")
            g.attrs["source"] = ep["source"]
            g.create_dataset("images", data=ep["images"], dtype="u1", compression="lzf")
            g.create_dataset("actions", data=ep["actions"], dtype="f4")
            g.create_dataset("states", data=ep["states"], dtype="f4")
            g.create_dataset("rewards", data=ep["rewards"], dtype="f4")
            g.create_dataset("dones", data=ep["dones"], dtype=bool)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[DONE] {len(episodes)} 条 / {total_frames} 帧 -> {args.out} ({size_mb:.1f} MB)",
          flush=True)


if __name__ == "__main__":
    main()
