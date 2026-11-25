#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""截取 PushCube 主观相机帧并保存可复现的初始状态（mujoco 环境）。"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")
from envs.push_cube_env import PushCubeEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="截取 PushCube 观测帧与初始状态")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "vla_single_step"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env = PushCubeEnv(obs_mode="both", rgb_preset="demo",
                      render_camera="global_cam", render_mode="rgb_array",
                      randomize_start=False)
    obs, info = env.reset(seed=args.seed)
    Image.fromarray(obs["image"]).save(os.path.join(args.out_dir, "observation_224.png"))
    Image.fromarray(env.render()).save(os.path.join(args.out_dir, "scene_before.png"))

    d = env.data
    np.savez(
        os.path.join(args.out_dir, "initial_state.npz"),
        qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(),
        mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy(),
        ctrl=d.ctrl.copy(), time=np.array(d.time),
    )
    meta = {
        "seed": args.seed,
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "target_xy": info["target_xy"].tolist(),
        "cube_start_xy": info["cube_start_xy"].tolist(),
        "ee_mocap_pos": d.mocap_pos[0].tolist(),
        "cube_dist": float(info["cube_target_dist"]),
    }
    with open(os.path.join(args.out_dir, "initial_state.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[CAPTURE] 观测尺寸:", obs["image"].shape, "动作范围:", meta["action_low"], meta["action_high"])
    print("[CAPTURE] 目标:", meta["target_xy"], "立方体起点:", meta["cube_start_xy"])
    env.close()


if __name__ == "__main__":
    main()
