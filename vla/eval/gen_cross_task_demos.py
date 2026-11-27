# -*- coding: utf-8 -*-
"""生成"推到左边 / 推到右边"跨任务的真实几何示教，组装成 few-shot 数据集。

- 用 envs.push_cube_env 的 scene_override 放置横向目标，几何控制器闭环推块；
- 每条轨迹记录 (image, state, action[4d 物理位移])；
- 只保留成功且步数最短的轨迹；左/右各取若干条，合计 --total 条（默认 5）；
- action 物理值按现有 norm_stats 归一化到 [-1,1]，组装成与主线同构的 HF Dataset，
  供 train.py --init-adapter 续训（few-shot）直接使用。

用法（mujoco 环境）：
  MUJOCO_GL=egl python vla/eval/gen_cross_task_demos.py \
    --out-dir results/vla_cross_task/fewshot_dataset --total 5

输出：
  <out-dir>/raw_cross/<split>/          HF Dataset（image/instruction/action）
  <out-dir>/cross_demos_meta.json       每条示教的目标、起点、成功率、步数
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MUJOCO_ROOT = Path("/data/wzh/robot/mujoco-studio-lab")
sys.path.insert(0, str(MUJOCO_ROOT))

import mujoco  # noqa: E402
from envs.push_cube_env import PushCubeEnv, HOME_POS  # noqa: E402

PUSH_Z = 0.86
FINGER_Q = 0.025
LIM = 0.02
N_APPROACH = 28
PUSH_MAX = 45
SUCCESS_DIST = 0.028

# 任务目标（横向 y 偏移）。左 = y 减小，右 = y 增大。均在桌面 Y ±0.30 与 mocap ±0.25 内。
TASKS = {
    "left": {
        "instruction": "把方块推到左边",
        "targets": [
            [0.80, -0.26], [0.78, -0.24], [0.82, -0.27], [0.79, -0.25],
        ],
    },
    "right": {
        "instruction": "把方块推到右边",
        "targets": [
            [0.80, 0.02], [0.78, 0.01], [0.82, 0.03], [0.79, 0.00],
        ],
    },
}
CUBE_START = [0.62, -0.12]


def run_geo_episode(env, target, seed):
    env.reset(seed=seed)
    scene = {"cube_xy": list(CUBE_START), "target_xy": list(target), "scene_id": seed}
    obs, info = env.reset(options={"scene": scene})
    env.data.qpos[7:9] = FINGER_Q
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    env._prev_dist = env._cube_target_dist()

    cube0 = np.array(CUBE_START, dtype=float)
    v = np.array(target, dtype=float) - cube0
    v = v / np.linalg.norm(v)
    rear = np.array([cube0[0] - v[0] * 0.09, cube0[1] - v[1] * 0.09, PUSH_Z])
    tgt = HOME_POS.copy()
    fr = np.cumsum(np.ones(N_APPROACH) / N_APPROACH)

    imgs, acts, sts = [], [], []
    for step in range(N_APPROACH + PUSH_MAX):
        imgs.append(obs["image"].copy())
        sts.append(obs["state"].copy())
        if step < N_APPROACH:
            goal = HOME_POS + (rear - HOME_POS) * fr[step]
            a3 = np.clip(goal - tgt, -LIM, LIM)
        else:
            cube = env.data.xpos[env._cube_body][:2]
            ta = np.array(target)
            if np.linalg.norm(cube - ta) < SUCCESS_DIST:
                a3 = np.zeros(3)
            else:
                d2 = np.clip(0.6 * (ta - cube), -LIM, LIM)
                a3 = np.array([d2[0], d2[1], 0.0])
        action = np.concatenate([a3, [0.0]]).astype(np.float32)
        acts.append(action)
        obs, r, term, trunc, info = env.step(action)
        tgt = env.data.mocap_pos[0].copy()
        if info.get("success"):
            break

    final = env.data.xpos[env._cube_body][:2]
    dist = float(np.linalg.norm(final - np.array(target)))
    return {
        "images": np.stack(imgs).astype(np.uint8),
        "actions": np.stack(acts).astype(np.float32),
        "states": np.stack(sts).astype(np.float32),
        "dist": dist,
        "steps": info["steps"],
        "success": bool(dist < 0.03),
    }


def normalize_actions(acts, norm):
    lo = np.asarray(norm["min"], dtype=np.float64)
    hi = np.asarray(norm["max"], dtype=np.float64)
    span = np.maximum(hi - lo, 1e-8)
    out = np.zeros_like(acts, dtype=np.float32)
    for j in range(4):
        if span[j] > 1e-8:
            out[:, j] = np.clip(2.0 * (acts[:, j] - lo[j]) / span[j] - 1.0, -1, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--total", type=int, default=5, help="few-shot 示教总条数（默认 5）")
    ap.add_argument("--per-task", type=int, default=0, help="每任务条数；0=按 total 均分")
    ap.add_argument("--norm-stats", default="/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset/norm_stats.json")
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm = json.load(open(args.norm_stats, encoding="utf-8"))["action"]

    per_task = args.per_task if args.per_task > 0 else max(1, args.total // 2)
    # 若 total 为奇数，多出来的一条给 right
    counts = {"left": per_task, "right": args.total - per_task}
    print(f"[fewshot] targets: left={counts['left']} right={counts['right']} total={args.total}", flush=True)

    env = PushCubeEnv(image_obs=True, obs_mode="both", scene_override=True,
                      max_episode_steps=N_APPROACH + PUSH_MAX, rgb_preset="demo")
    all_rows = []
    meta = {"task": "push_cube_left_right", "total": args.total,
            "counts": counts, "episodes": [], "norm_stats": norm}

    rng = np.random.default_rng(args.seed)
    for task, cfg in TASKS.items():
        candidates = []
        for ti, target in enumerate(cfg["targets"]):
            for rep in range(2):  # 每个目标 2 次不同初始扰动
                seed = args.seed + ti * 10 + rep
                ep = run_geo_episode(env, target, seed)
                candidates.append((target, ep))
                print(f"[fewshot] {task} target={target} rep={rep} "
                      f"dist={ep['dist']:.3f} steps={ep['steps']} succ={ep['success']}", flush=True)
        cand_ok = [c for c in candidates if c[1]["success"]]
        cand_ok.sort(key=lambda c: (c[1]["dist"], c[1]["steps"]))
        chosen = cand_ok[:counts[task]]
        if len(chosen) < counts[task]:
            raise RuntimeError(f"{task}: only {len(chosen)} successful demos, need {counts[task]}")
        for target, ep in chosen:
            for i in range(len(ep["actions"])):
                all_rows.append({
                    "image": ep["images"][i],
                    "instruction": cfg["instruction"],
                    "action": ep["actions"][i],
                })
            meta["episodes"].append({
                "task": task, "instruction": cfg["instruction"],
                "target_xy": target, "cube_start_xy": CUBE_START,
                "dist": round(ep["dist"], 5), "steps": ep["steps"],
                "success": ep["success"], "n_frames": len(ep["actions"]),
            })
    env.close()
    print(f"[fewshot] collected {len(all_rows)} frames from {sum(counts.values())} demos", flush=True)

    # 保存原始示教 npz（mujoco 环境无 datasets，组装交给 vla 环境的 build 脚本）
    np.savez_compressed(
        str(out_dir / "cross_demos_raw.npz"),
        images=np.stack([r["image"] for r in all_rows]).astype(np.uint8),
        actions=np.stack([r["action"] for r in all_rows]).astype(np.float32),
        instructions=np.array([r["instruction"] for r in all_rows]),
        lengths=np.array([e["n_frames"] for e in meta["episodes"]], dtype=np.int64),
    )
    with open(out_dir / "cross_demos_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[fewshot] raw npz -> {out_dir / 'cross_demos_raw.npz'}  {len(all_rows)} frames", flush=True)
    print("CROSS_DEMOS_DONE", flush=True)


if __name__ == "__main__":
    main()
