#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""采集左右目标变体轨迹（mujoco 环境运行）。

复用既有 scripts/collect_vla_demos.py 的脚本化专家控制律，唯一改动是把目标
红盘位置通过 PushCubeEnv(scene_override=True) 注入为变体 target_xy，从而产生
「推到目标区偏左 / 偏右」两类新轨迹。图像/动作/状态字段口径与基础数据一致，
供后续统一归一化与格式转换。

数据落在仓库外，不入 git。输出 variants/{left,right}.npz + manifest.json。

用法（mujoco 环境，EGL 离屏渲染）:
  MUJOCO_GL=egl $MUJOCO python vla_finetune/data/collect_variants.py \
      --episodes 160 --select 10 --seed 20261126 \
      --out /data/wzh/robot/datasets/vla_pushcube_finetune
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MUJOCO_REPO = "/data/wzh/robot/mujoco-studio-lab"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 与基础采集一致的随机化/筛选参数
CUBE_XY_RANGE = (-0.05, 0.05)
ARM_XYZ_RANGES = ((-0.010, 0.010), (-0.010, 0.010), (-0.006, 0.006))
PUSH_STEPS = 34
SUCCESS_DIST = 0.028
MIN_FRAMES, MAX_FRAMES = 30, 40
MAX_CUBE_STEP_DISP = 0.03
CUBE_Z_MIN, CUBE_Z_MAX = 0.76, 0.82
# 变体立方体起点 y 贴近各自目标 y（±0.03），保证横向修正幅度可控、成功率可用；
# 红盘目标位置仍真实改变，场景与语言差异保留。
CUBE_Y_START_JITTER = (-0.03, 0.03)


def collect_one_variant(name, cfg, episodes, select, seed, out_dir, env, cd, fps):
    rng = np.random.default_rng(seed + (0 if name == "left" else 1000))
    target_xy = np.asarray(cfg["target_xy"], dtype=np.float64)
    records = []
    rejects = {"not_success": 0, "frame_range": 0, "tunneling": 0, "reached_target_xy": 0}
    instr = cfg["instructions"][0]  # 主指令；同义表述在 build_hf_dataset 阶段展开

    for ep in range(episodes):
        cube_off = np.array([rng.uniform(*CUBE_XY_RANGE),
                             rng.uniform(*CUBE_Y_START_JITTER)])
        arm_off = np.array([rng.uniform(*r) for r in ARM_XYZ_RANGES])
        cube_start = np.array([0.62, float(target_xy[1])]) + cube_off
        scene = {"cube_xy": cube_start.tolist(), "target_xy": target_xy.tolist(),
                 "scene_id": 0}
        obs, _ = env.reset(seed=5000 + ep, options={"scene": scene})

        # reset 后 _apply_scene 已把立方体放到 cube_start、红盘放到 target_xy
        cube_start_actual = env.data.xpos[env._cube_body][:3].copy()
        env.data.qpos[7:9] = cd.FINGER_Q
        start_pos = cd.HOME_POS + arm_off
        env.data.mocap_pos[0] = start_pos
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        env._prev_dist = env._cube_target_dist()

        lane_y = float(cube_start_actual[1])
        rear = np.array([cube_start_actual[0] - 0.08, lane_y, cd.PUSH_Z])
        fracs = cd.ease_frac(cd.APPROACH_STEPS)

        ep_imgs, ep_acts, ep_sts, ep_ts = [], [], [], []
        ep_cube_xy, ep_cube_z = [], []
        tgt = start_pos.copy()
        hold = 0
        for step in range(cd.APPROACH_STEPS + PUSH_STEPS + cd.HOLD_STEPS):
            obs_t = obs
            if step < cd.APPROACH_STEPS:
                goal = start_pos + (rear - start_pos) * fracs[step]
                a3 = np.clip(goal - tgt, -cd.STEP_MAX, cd.STEP_MAX)
            else:
                cube_xy = env.data.xpos[env._cube_body][:2]
                if float(np.linalg.norm(cube_xy - target_xy)) < cd.SUCCESS_DIST:
                    hold += 1
                    a3 = np.zeros(3)
                    if hold >= cd.HOLD_STEPS:
                        break
                else:
                    dy_err = target_xy[1] - cube_xy[1]
                    dy = float(np.clip(0.9 * dy_err, -0.008, 0.008))
                    dx = float(np.clip(cd.STEP_MAX, 0.0,
                                       cd.MOCAP_HIGH[0] - env.data.mocap_pos[0][0]))
                    a3 = np.array([dx, dy, 0.0])
            action = np.concatenate([a3, [0.0]]).astype(np.float32)
            obs, _r, _t, _tr, _info = env.step(action)
            ep_imgs.append(obs_t["image"].copy())
            ep_acts.append(action)
            ep_sts.append(obs_t["state"].copy())
            ep_ts.append(step / fps)
            cpos = env.data.xpos[env._cube_body]
            ep_cube_xy.append(cpos[:2].copy())
            ep_cube_z.append(float(cpos[2]))
            tgt = env.data.mocap_pos[0].copy()

        n = len(ep_acts)
        final = env.data.xpos[env._cube_body]
        dist = float(np.linalg.norm(final[:2] - target_xy))
        success = dist < SUCCESS_DIST
        xy = np.asarray(ep_cube_xy)
        max_disp = float(np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if n > 1 else 0.0
        zmin, zmax = float(np.min(ep_cube_z)), float(np.max(ep_cube_z))
        tunneling = (max_disp >= MAX_CUBE_STEP_DISP) or (zmin < CUBE_Z_MIN) or (zmax > CUBE_Z_MAX)

        if not success:
            rejects["not_success"] += 1
        elif not (MIN_FRAMES <= n <= MAX_FRAMES):
            rejects["frame_range"] += 1
        elif tunneling:
            rejects["tunneling"] += 1
        else:
            records.append({
                "imgs": ep_imgs, "acts": ep_acts, "sts": ep_sts, "ts": ep_ts,
                "steps": n, "dist": dist, "start": cube_start_actual,
                "max_disp": max_disp, "zmin": zmin, "zmax": zmax,
                "cube_off": cube_off, "arm_off": arm_off,
            })
        if (ep + 1) % 20 == 0:
            print(f"[variant-{name}] ep{ep:03d}: kept={len(records)} dist={dist:.3f} "
                  f"n={n} succ={success} tunnel={tunneling}")
        if len(records) >= select:
            break

    selected = records[:select]
    if not selected:
        raise RuntimeError(f"variant {name}: no qualified trajectory; rejects={rejects}")
    images = np.stack([f for r in selected for f in r["imgs"]]).astype(np.uint8)
    actions = np.stack([a for r in selected for a in r["acts"]]).astype(np.float32)
    states = np.stack([s for r in selected for s in r["sts"]]).astype(np.float32)
    timestamps = np.concatenate([np.asarray(r["ts"], dtype=np.float32) for r in selected])
    episode_lengths = np.array([r["steps"] for r in selected], dtype=np.int64)
    ep_dist = np.array([r["dist"] for r in selected], dtype=np.float32)

    npz_path = os.path.join(out_dir, f"variant_{name}.npz")
    np.savez_compressed(
        npz_path,
        images=images, actions=actions, states=states, timestamps=timestamps,
        episode_lengths=episode_lengths,
        episode_instructions=np.array([instr] * len(selected), dtype=object),
        episode_success=np.array([True] * len(selected)),
        episode_dist=ep_dist,
        target_xy=np.tile(target_xy, (len(selected), 1)).astype(np.float32),
        variant=np.array([name] * len(selected), dtype=object),
        fps=np.int64(fps),
    )
    info = {
        "variant": name, "category": cfg["category"], "target_xy": cfg["target_xy"],
        "instructions": cfg["instructions"], "count": len(selected),
        "total_frames": int(len(images)), "fps": int(fps),
        "tried": ep + 1, "rejects": rejects,
        "final_dist_max": round(float(ep_dist.max()), 5),
        "frames_min_max": [int(episode_lengths.min()), int(episode_lengths.max())],
    }
    print(f"[variant-{name}] saved {len(images)} frames / {len(selected)} eps -> {npz_path}")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=160)
    ap.add_argument("--select", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--out", default="/data/wzh/robot/datasets/vla_pushcube_finetune")
    args = ap.parse_args()

    cd = _load_module("_vcd", os.path.join(MUJOCO_REPO, "scripts", "collect_demos.py"))
    iv = _load_module("_viv", os.path.join(REPO, "vla_finetune", "data", "instruction_variants.py"))

    import mujoco  # noqa: F401
    if MUJOCO_REPO not in sys.path:
        sys.path.insert(0, MUJOCO_REPO)
    from envs.push_cube_env import PushCubeEnv, HOME_POS  # noqa: E402
    cd.HOME_POS = HOME_POS  # 专家控制律常量补齐（原模块从 env 导入）

    env = PushCubeEnv(image_obs=True, max_episode_steps=200, scene_override=True)
    fps = int(round(1.0 / (env.model.opt.timestep * env.frame_skip)))
    out_dir = os.path.join(args.out, "variants")
    os.makedirs(out_dir, exist_ok=True)

    summary = {"source": "PushCubeEnv scene_override target_xy",
               "fps": fps, "per_variant": []}
    with open(os.path.join(REPO, "vla_finetune", "data", "instruction_variants.py"), encoding="utf-8") as f:
        pass  # 校验文件存在
    for name, cfg in iv.VARIANTS.items():
        info = collect_one_variant(name, cfg, args.episodes, args.select,
                                   args.seed, out_dir, env, cd, fps)
        summary["per_variant"].append(info)
    env.close()

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[完成] 变体采集 manifest 已保存")


if __name__ == "__main__":
    main()
