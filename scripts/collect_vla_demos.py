#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VLA 数据集专用专家轨迹采集（mujoco 环境运行）。

复用既有 scripts/collect_demos_v3.py 的脚本化专家控制律（接近 + 闭环推 +
到位保持），在其基础上做本阶段要求的扩展：

  * 立方体初始位置 x/y 均 ±0.05 m 随机化（v3 是 x±0.035 / y±0.055）；
  * 机械臂初始位姿 xyz 随机扰动；
  * 每条轨迹记录回合内时间戳 timestamps（帧序/fps）；
  * 逐轨迹分配指令模板 id 与英文指令（来自 vla/data/instruction_templates.py）；
  * 更严格质量筛选：到达目标（dist<0.03）、帧数 30–40、无穿模/瞬移
    （相邻帧立方体平面位移 < 0.03 m 且 z 在桌面合理区间）。

数据落在仓库外（默认 /data/wzh/robot/datasets/vla_pushcube_25/），不入 git。
输出：raw_demos_vla.npz + instruction_manifest.json + quality_report.json。

用法（mujoco 环境，EGL 离屏渲染）：
  MUJOCO_GL=egl $MUJOCO scripts/collect_vla_demos.py \
      --episodes 160 --select 25 --seed 20261125 \
      --out /data/wzh/robot/datasets/vla_pushcube_25
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MUJOCO_REPO = "/data/wzh/robot/mujoco-studio-lab"

# 复用既有采集模块的控制常量与缓动函数（按文件路径加载，避免包遮蔽）
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- 本阶段随机化与筛选参数 -------------------------------------------------
CUBE_XY_RANGE = (-0.05, 0.05)          # 立方体 x/y 初始偏移（m）
ARM_XYZ_RANGES = ((-0.010, 0.010), (-0.010, 0.010), (-0.006, 0.006))
PUSH_STEPS = 34                        # 推的预算（成功会提前 break）
SUCCESS_DIST = 0.03
MIN_FRAMES, MAX_FRAMES = 30, 40
MAX_CUBE_STEP_DISP = 0.03              # 相邻帧立方体平面位移上限（穿模/瞬移判据）
CUBE_Z_MIN, CUBE_Z_MAX = 0.76, 0.82    # 桌面高度 0.75 + 立方体半高附近


def collect(episodes, select, seed, out_dir):
    cd = _load_module("_vla_cd", os.path.join(MUJOCO_REPO, "scripts", "collect_demos.py"))
    itpl = _load_module("_vla_itpl", os.path.join(REPO, "vla", "data", "instruction_templates.py"))

    import mujoco
    # push_cube_env.py 用相对导入 .offscreen_render，须作为 envs 包导入
    if MUJOCO_REPO not in sys.path:
        sys.path.insert(0, MUJOCO_REPO)
    from envs.push_cube_env import PushCubeEnv, HOME_POS, TARGET_XY  # noqa: E402

    env = PushCubeEnv(image_obs=True, max_episode_steps=200)
    fps = int(round(1.0 / (env.model.opt.timestep * env.frame_skip)))
    print(f"[vla-gen] cube xy{CUBE_XY_RANGE} arm jitter xyz; dt={env.model.opt.timestep} fps={fps}")

    rng = np.random.default_rng(seed)
    instr_ids = itpl.sequence_for_count(select, seed=seed)
    records = []
    rejects = {"not_success": 0, "frame_range": 0, "tunneling": 0}
    target_xy = np.asarray(TARGET_XY, dtype=np.float64)

    for ep in range(episodes):
        cube_off = np.array([rng.uniform(*CUBE_XY_RANGE), rng.uniform(*CUBE_XY_RANGE)])
        arm_off = np.array([rng.uniform(*r) for r in ARM_XYZ_RANGES])
        obs, _ = env.reset(seed=3000 + ep)
        cube_start = env.data.xpos[env._cube_body][:3].copy()
        cube_start[0] += cube_off[0]
        cube_start[1] += cube_off[1]
        env.data.qpos[9:12] = cube_start
        env.data.qvel[9:15] = 0.0
        env.data.qpos[7:9] = cd.FINGER_Q
        start_pos = HOME_POS + arm_off
        env.data.mocap_pos[0] = start_pos
        mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        env._prev_dist = env._cube_target_dist()

        lane_y = float(cube_start[1])
        rear = np.array([cube_start[0] - 0.08, lane_y, cd.PUSH_Z])
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
                    dy = float(np.clip(0.7 * dy_err, -0.005, 0.005))
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

        # ---- 筛选 ----
        if not success:
            rejects["not_success"] += 1
        elif not (MIN_FRAMES <= n <= MAX_FRAMES):
            rejects["frame_range"] += 1
        elif tunneling:
            rejects["tunneling"] += 1
        else:
            records.append({
                "imgs": ep_imgs, "acts": ep_acts, "sts": ep_sts,
                "ts": ep_ts, "steps": n, "dist": dist,
                "start": cube_start, "arm_start": start_pos,
                "max_disp": max_disp, "zmin": zmin, "zmax": zmax,
                "cube_off": cube_off, "arm_off": arm_off,
            })
        if (ep + 1) % 20 == 0:
            print(f"[vla-gen] ep{ep:03d}: kept={len(records)} dist={dist:.3f} "
                  f"n={n} succ={success} tunnel={tunneling}")
        if len(records) >= select:
            break
    env.close()

    if len(records) < select:
        print(f"[vla-gen] WARN only {len(records)}/{select} qualified from {episodes} tries; rejects={rejects}")
    selected = records[:select]
    if not selected:
        raise RuntimeError("no qualified trajectory; relax filters or increase --episodes")

    os.makedirs(out_dir, exist_ok=True)
    n_img = sum(len(r["imgs"]) for r in selected)
    images = np.stack([f for r in selected for f in r["imgs"]]).astype(np.uint8)
    actions = np.stack([a for r in selected for a in r["acts"]]).astype(np.float32)
    states = np.stack([s for r in selected for s in r["sts"]]).astype(np.float32)
    timestamps = np.concatenate([np.asarray(r["ts"], dtype=np.float32) for r in selected])
    episode_lengths = np.array([r["steps"] for r in selected], dtype=np.int64)
    cube_starts = np.stack([r["start"] for r in selected]).astype(np.float32)
    arm_starts = np.stack([r["arm_start"] for r in selected]).astype(np.float32)
    ep_success = np.array([True] * len(selected))
    ep_dist = np.array([r["dist"] for r in selected], dtype=np.float32)
    ep_instr = np.array([itpl.get_instruction(tid) for tid in instr_ids[:len(selected)]], dtype=object)
    ep_tids = np.array(instr_ids[:len(selected)], dtype=object)

    npz_path = os.path.join(out_dir, "raw_demos_vla.npz")
    np.savez_compressed(
        npz_path,
        images=images, actions=actions, states=states,
        timestamps=timestamps, episode_lengths=episode_lengths,
        cube_starts=cube_starts, arm_starts=arm_starts,
        episode_instructions=ep_instr, episode_template_ids=ep_tids,
        episode_success=ep_success, episode_dist=ep_dist,
        fps=np.int64(fps), task=cd.TASK,
    )

    manifest = {
        "count": len(selected), "total_frames": n_img, "fps": fps,
        "instruction_source": "vla/data/instruction_templates.py",
        "categories": itpl.CATEGORY_DESC,
        "assignments": [
            {"episode": i, "template_id": ep_tids[i],
             "category": itpl.get_category(ep_tids[i]), "instruction": str(ep_instr[i]),
             "frames": int(episode_lengths[i]), "final_dist": round(float(ep_dist[i]), 5)}
            for i in range(len(selected))
        ],
    }
    with open(os.path.join(out_dir, "instruction_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    quality = {
        "tried": ep + 1, "kept": len(selected), "rejects": rejects,
        "filters": {
            "success_dist": SUCCESS_DIST, "frame_range": [MIN_FRAMES, MAX_FRAMES],
            "max_adjacent_cube_disp": MAX_CUBE_STEP_DISP,
            "cube_z_range": [CUBE_Z_MIN, CUBE_Z_MAX],
        },
        "episodes": [
            {"episode": i, "frames": int(r["steps"]), "final_dist": round(r["dist"], 5),
             "cube_offset": [round(float(x), 4) for x in r["cube_off"]],
             "arm_offset": [round(float(x), 4) for x in r["arm_off"]],
             "max_adjacent_disp": round(r["max_disp"], 5),
             "cube_z": [round(r["zmin"], 4), round(r["zmax"], 4)]}
            for i, r in enumerate(selected)
        ],
    }
    with open(os.path.join(out_dir, "quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)

    lens = episode_lengths.tolist()
    print(f"[vla-gen] saved {n_img} frames / {len(selected)} episodes -> {npz_path}")
    print(f"[vla-gen] frames min/max/mean = {min(lens)}/{max(lens)}/{np.mean(lens):.1f}; "
          f"dist max = {float(ep_dist.max()):.4f}; rejects={rejects}")
    return npz_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=160)
    p.add_argument("--select", type=int, default=25)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--out", default="/data/wzh/robot/datasets/vla_pushcube_25")
    args = p.parse_args()
    collect(args.episodes, args.select, args.seed, args.out)


if __name__ == "__main__":
    main()
