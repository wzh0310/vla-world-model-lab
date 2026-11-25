#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenVLA 闭环推理：环境读图 → VLA 推理 → 动作映射 → 环境步进，跑完整回合。

在 vla 进程内注入 mujoco 环境 site-packages，使模型只加载一次、环境全程驻留。
支持多条指令、逐步记录、可选诊断（通道顺序/方向反转/推理速度），结果落盘。

用法（vla 环境）：
    python scripts/rollout_vla_closed_loop.py --instructions "push the cube to the red target" \
        --max-steps 30 --out-dir results/vla_closed_loop --quant none
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# 按文件路径加载适配器：mujoco-studio-lab 也有同名 utils 包，避免包名遮蔽
_adapter_path = os.path.join(ROOT, "vla", "utils", "vla_adapter.py")
_spec = importlib.util.spec_from_file_location("vla_adapter", _adapter_path)
vla_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vla_adapter)
OpenVLAPolicy = vla_adapter.OpenVLAPolicy
parse_action = vla_adapter.parse_action

# 注入 mujoco 环境的依赖（gymnasium/mujoco），vla 进程模型只加载一次
MUJOCO_SITE = "/opt/miniconda3/envs/mujoco/lib/python3.12/site-packages"
if os.path.isdir(MUJOCO_SITE) and MUJOCO_SITE not in sys.path:
    sys.path.append(MUJOCO_SITE)
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")

from envs.push_cube_env import PushCubeEnv  # noqa: E402


DEFAULT_INSTRUCTIONS = [
    "push the cube to the red target",
    "move the cube to the red circle",
    "push the block onto the red goal",
]


def parse_args():
    ap = argparse.ArgumentParser(description="OpenVLA 闭环回合推理")
    ap.add_argument("--model-dir", default="/data/wzh/robot/models/openvla-7b")
    ap.add_argument("--quant", choices=("none", "int4"), default="none")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--instructions", nargs="+", default=DEFAULT_INSTRUCTIONS,
                    help="一条或多条英文指令，每条跑一个完整回合")
    ap.add_argument("--max-steps", type=int, default=30, help="每回合最大步数")
    ap.add_argument("--scale", type=float, default=0.02, help="位置增量映射上限（米）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "vla_closed_loop"))
    ap.add_argument("--save-frames", action="store_true", help="保存每步环境截图")
    ap.add_argument("--diagnose", action="store_true",
                    help="额外跑通道顺序/方向反转诊断（每条指令首步）")
    return ap.parse_args()


def make_env(args):
    return PushCubeEnv(
        obs_mode="both", rgb_preset="demo", render_camera="global_cam",
        render_mode="rgb_array", randomize_start=False,
        max_episode_steps=args.max_steps,
    )


def run_episode(env, policy, instruction, args, ep_idx):
    """跑一个完整回合，返回逐步记录与汇总指标。"""
    out_dir = os.path.join(args.out_dir, f"ep{ep_idx:02d}")
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    obs, info = env.reset(seed=args.seed + ep_idx)
    target = np.array(info["target_xy"], dtype=float)
    dist0 = float(info["cube_target_dist"])
    records = []
    latencies = []
    success = False
    t_ep = time.perf_counter()

    if args.save_frames:
        Image.fromarray(env.render()).save(os.path.join(frames_dir, "step_0000.png"))

    for step in range(args.max_steps):
        # 1) 环境读图像（主观相机 224×224 RGB）
        rgb = obs["image"]
        # 2) VLA 推理 + 3) 动作映射
        rec = policy.predict(rgb, instruction, scale=args.scale)
        mapped = rec["mapped_4d"]
        latencies.append(rec["latency_s"])

        # 诊断：仅首步做对照，不把诊断动作真正写进轨迹
        if args.diagnose and step == 0:
            bgr = policy.predict(rgb, instruction, scale=args.scale, channel_mode="bgr")
            flip = policy.predict(rgb, instruction, scale=args.scale, flip_xyz=True)
            rec["diagnosis"] = {
                "bgr_mapped_4d": bgr["mapped_4d"],
                "bgr_delta_vs_rgb": [round(bgr["mapped_4d"][i] - mapped[i], 6)
                                     for i in range(3)],
                "flip_mapped_4d": flip["mapped_4d"],
                "flip_delta_vs_rgb": [round(flip["mapped_4d"][i] - mapped[i], 6)
                                      for i in range(3)],
            }

        # 4) 环境步进
        ee_before = env.data.mocap_pos[0].copy()
        obs, reward, term, trunc, info = env.step(np.asarray(mapped, dtype=np.float64))
        ee_after = env.data.mocap_pos[0].copy()
        cube_xy = env.data.xpos[env._cube_body][:2].copy()
        dist = float(info["cube_target_dist"])
        success = bool(info.get("success", False))

        step_rec = {
            "step": step + 1,
            "instruction": instruction,
            "mapped_4d": mapped,
            "action_parsed": parse_action(mapped),
            "ee_delta": [round(float(v), 6) for v in (ee_after - ee_before)],
            "cube_xy": [round(float(v), 6) for v in cube_xy],
            "dist_to_target": round(dist, 6),
            "dist_delta": round(dist - (records[-1]["dist_to_target"] if records else dist0), 6),
            "reward": round(float(reward), 6),
            "latency_s": rec["latency_s"],
            "vram_reserved_gb": rec["vram_reserved_gb"],
        }
        if "diagnosis" in rec:
            step_rec["diagnosis"] = rec["diagnosis"]
        records.append(step_rec)

        if args.save_frames and ((step + 1) % 5 == 0 or term or trunc):
            Image.fromarray(env.render()).save(
                os.path.join(frames_dir, f"step_{step + 1:04d}.png"))
        if term or trunc:
            break

    wall = time.perf_counter() - t_ep
    dists = [r["dist_to_target"] for r in records]
    summary = {
        "instruction": instruction,
        "seed": args.seed + ep_idx,
        "target_xy": [round(float(v), 6) for v in target],
        "dist_start": round(dist0, 6),
        "dist_end": round(dists[-1] if dists else dist0, 6),
        "dist_min": round(min(dists) if dists else dist0, 6),
        "dist_improvement": round(dist0 - (dists[-1] if dists else dist0), 6),
        "success": success,
        "steps": len(records),
        "total_reward": round(float(sum(r["reward"] for r in records)), 4),
        "latency_mean_s": round(float(np.mean(latencies)), 4) if latencies else None,
        "latency_max_s": round(float(np.max(latencies)), 4) if latencies else None,
        "wall_time_s": round(wall, 2),
        "vram_reserved_gb": policy.memory_gb()["reserved"],
    }
    with open(os.path.join(out_dir, "episode.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "steps": records}, f, ensure_ascii=False, indent=2)

    print(f"[EPISODE {ep_idx}] 指令='{instruction}' 步数={len(records)} "
          f"距离 {dist0:.4f}→{summary['dist_end']:.4f} (最小 {summary['dist_min']:.4f}) "
          f"success={success} reward={summary['total_reward']:.2f} "
          f"耗时={wall:.1f}s", flush=True)
    return summary


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[闭环] 指令数={len(args.instructions)} 每回合步数={args.max_steps} "
          f"精度={args.quant} 映射量程=±{args.scale}m", flush=True)
    t0 = time.perf_counter()
    policy = OpenVLAPolicy(
        model_dir=args.model_dir, quant=args.quant, device=args.device,
        unnorm_key=args.unnorm_key,
    )
    print(f"[闭环] 模型加载完成，参数量={policy.n_params / 1e9:.2f}B "
          f"显存={policy.memory_gb()} 耗时={time.perf_counter() - t0:.1f}s", flush=True)

    env = make_env(args)
    summaries = []
    for i, instr in enumerate(args.instructions):
        summaries.append(run_episode(env, policy, instr, args, i))
    env.close()

    overall = {
        "config": {
            "quant": args.quant, "unnorm_key": args.unnorm_key,
            "scale_m": args.scale, "max_steps": args.max_steps,
            "seed": args.seed, "n_instructions": len(args.instructions),
            "model_dir": args.model_dir,
        },
        "episodes": summaries,
        "success_rate": round(sum(s["success"] for s in summaries) / max(len(summaries), 1), 4),
        "mean_latency_s": round(float(np.mean([s["latency_mean_s"] for s in summaries
                                                if s["latency_mean_s"] is not None])), 4),
        "total_wall_s": round(time.perf_counter() - t0, 2),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"[闭环] 全部完成：成功率={overall['success_rate']} "
          f"平均单步耗时={overall['mean_latency_s']}s "
          f"总耗时={overall['total_wall_s']}s", flush=True)
    print(f"[闭环] 结果目录: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
