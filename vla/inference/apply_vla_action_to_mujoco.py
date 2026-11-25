#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 VLA 映射后的动作在 PushCube 真实环境执行一步（mujoco 环境）。"""
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
    ap = argparse.ArgumentParser(description="在 PushCube 中执行 VLA 单步动作")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "vla_single_step"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    state = np.load(os.path.join(args.out_dir, "initial_state.npz"))
    with open(os.path.join(args.out_dir, "vla_action.json"), encoding="utf-8") as f:
        act_rec = json.load(f)
    action = np.asarray(act_rec["mapped_4d"], dtype=np.float64)

    env = PushCubeEnv(obs_mode="both", rgb_preset="demo",
                      render_camera="global_cam", render_mode="rgb_array",
                      randomize_start=False)
    env.reset(seed=args.seed)
    d = env.data
    d.qpos[:] = state["qpos"]
    d.qvel[:] = state["qvel"]
    d.act[:] = state["act"]
    d.mocap_pos[:] = state["mocap_pos"]
    d.mocap_quat[:] = state["mocap_quat"]
    d.ctrl[:] = state["ctrl"]
    d.time = float(state["time"])
    mujoco.mj_forward(env.model, d)
    env._steps = 0
    env._prev_dist = env._cube_target_dist()
    env._rgb_cache = None

    ee_before = d.mocap_pos[0].copy()
    cube_before = d.xpos[env._cube_body][:2].copy()
    dist_before = float(env._cube_target_dist())
    Image.fromarray(env.render()).save(os.path.join(args.out_dir, "scene_before_step.png"))
    Image.fromarray(env._get_obs()["image"]).save(os.path.join(args.out_dir, "obs_before_step.png"))

    obs, reward, term, trunc, info = env.step(action)
    ee_after = d.mocap_pos[0].copy()
    cube_after = d.xpos[env._cube_body][:2].copy()
    dist_after = float(info["cube_target_dist"])
    Image.fromarray(env.render()).save(os.path.join(args.out_dir, "scene_after_step.png"))
    Image.fromarray(obs["image"]).save(os.path.join(args.out_dir, "obs_after_step.png"))

    target = env.target_xy
    rec = {
        "action": action.tolist(),
        "ee_delta": (ee_after - ee_before).round(6).tolist(),
        "cube_delta_xy": (cube_after - cube_before).round(6).tolist(),
        "cube_dist_before": round(dist_before, 6),
        "cube_dist_after": round(dist_after, 6),
        "dist_change": round(dist_after - dist_before, 6),
        "reward": round(float(reward), 6),
        "terminated": bool(term), "truncated": bool(trunc),
        "target_xy": target.tolist(),
    }
    with open(os.path.join(args.out_dir, "mujoco_step_result.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("[MUJOCO] 执行动作:", np.round(action, 5).tolist())
    print("[MUJOCO] EE 位移:", rec["ee_delta"], "立方体位移:", rec["cube_delta_xy"])
    print("[MUJOCO] 目标距离:", rec["cube_dist_before"], "->", rec["cube_dist_after"],
          "reward:", rec["reward"])
    print("[MUJOCO] 结果已保存:", os.path.join(args.out_dir, "mujoco_step_result.json"))
    env.close()


if __name__ == "__main__":
    main()
