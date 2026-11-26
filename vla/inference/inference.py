#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenVLA 单样本推理 + PushCube 环境验证（vla 环境）。

输入：单张 224×224 RGB 图像（文件或实时环境观测）+ 自然语言指令。
输出：OpenVLA 原始 7 维动作 → 反归一化 → PushCube 4 维物理动作，
      并在真实 MuJoCo 场景中执行一步，校验动作幅度与方向合理性。

默认用 int4（NF4）量化加载，单卡显存约 4.6 GB（≤8 GB）；
可用 --quant none 切回 bf16（约 15.5 GB）做对照。

注意：实时取图路径必须设置 MUJOCO_GL=egl，否则 MuJoCo 离屏渲染不可用、
      obs["image"] 会退化为全零占位。脚本对此做了显式拦截。

用法（vla 环境）：
    # 用已保存的单帧图像 + 初始状态复现
    python vla/inference/inference.py \
        --image results/vla_single_step/observation_224.png \
        --state results/vla_single_step/initial_state.npz \
        --instruction "push the cube to the red target"

    # 直接从实时环境取一帧推理
    python vla/inference/inference.py --instruction "push the cube to the red target"

结果落盘到 --out（默认 results/vla_inference_single/inference_result.json）。
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
_ADAPTER_PATH = os.path.join(ROOT, "vla", "utils", "vla_adapter.py")
_spec = importlib.util.spec_from_file_location("vla_adapter", _ADAPTER_PATH)
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)
OpenVLAPolicy = _adapter.OpenVLAPolicy
parse_action = _adapter.parse_action

# 在 vla 进程内注入 mujoco 环境的依赖，模型只加载一次、环境驻留
MUJOCO_SITE = "/opt/miniconda3/envs/mujoco/lib/python3.12/site-packages"
if os.path.isdir(MUJOCO_SITE) and MUJOCO_SITE not in sys.path:
    sys.path.append(MUJOCO_SITE)
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")

import mujoco  # noqa: E402
from envs.push_cube_env import PushCubeEnv  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="OpenVLA 单样本推理 + MuJoCo 动作验证")
    ap.add_argument("--model-dir", default="/data/wzh/robot/models/openvla-7b")
    ap.add_argument("--quant", choices=("int4", "none"), default="int4",
                    help="int4=NF4 4bit（≤8GB）；none=bf16（约 15.5GB）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--unnorm-key", default="bridge_orig",
                    help="必须与模型 config.norm_stats 中的键一致")
    ap.add_argument("--image", default=None,
                    help="单帧图像路径；不传则直接从实时环境取一帧")
    ap.add_argument("--state", default=None,
                    help="与 --image 对应的环境初始状态 npz；不传则用新 reset 的状态")
    ap.add_argument("--instruction", default="push the cube to the red target")
    ap.add_argument("--scale", type=float, default=0.02, help="位置增量上限（米）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    ap.add_argument("--no-env-step", action="store_true",
                    help="只输出动作、不执行环境步进（纯推理验证）")
    return ap.parse_args()


def restore_state(env, state_path):
    """把环境恢复到与输入图像对应的初始状态。"""
    if not state_path:
        return None
    state = np.load(state_path)
    d = env.data
    for key in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "ctrl"):
        if key in state:
            getattr(d, key)[:] = state[key]
    if "time" in state:
        d.time = float(state["time"])
    mujoco.mj_forward(env.model, d)
    env._steps = 0
    env._prev_dist = env._cube_target_dist()
    env._rgb_cache = None
    return {k: list(np.asarray(state[k]).shape) for k in state.files}


def main():
    args = parse_args()
    out_path = args.out or os.path.join(ROOT, "results", "vla_inference_single",
                                        "inference_result.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- 1) 加载模型（int4 默认，显存 ≤8GB） ----
    t_load = time.perf_counter()
    policy = OpenVLAPolicy(
        model_dir=args.model_dir, quant=args.quant, device=args.device,
        unnorm_key=args.unnorm_key,
    )
    load_s = time.perf_counter() - t_load

    # ---- 2) 准备单帧图像与环境 ----
    env = PushCubeEnv(
        obs_mode="both", rgb_preset="demo", render_camera="global_cam",
        render_mode="rgb_array", randomize_start=False,
    )
    obs, info = env.reset(seed=args.seed)
    state_shapes = restore_state(env, args.state)
    target_xy = np.asarray(env.target_xy, dtype=float)
    cube_xy = env.data.xpos[env._cube_body][:2].copy()
    dist0 = float(env._cube_target_dist())

    if args.image:
        pil = Image.open(args.image).convert("RGB")
        image_source = args.image
    else:
        raw = np.asarray(obs["image"])
        # 离屏渲染未初始化时 obs["image"] 会退化为 shape=(1,) 的全零占位，
        # 必须显式拦截，否则会把空白图喂给模型得到无意义结果。
        if raw.ndim != 3 or raw.shape != (224, 224, 3) or float(raw.max()) <= 0:
            raise RuntimeError(
                "实时观测不是有效 224x224 RGB 图像（shape=%s, max=%s）。"
                "请先设置 MUJOCO_GL=egl 再运行，例如：MUJOCO_GL=egl python %s ..."
                % (raw.shape, raw.max() if raw.size else None, os.path.basename(__file__))
            )
        pil = Image.fromarray(raw)
        image_source = "live_env"
        pil.save(os.path.join(os.path.dirname(out_path), "observation_224.png"))

    # ---- 3) 单样本推理 + 动作映射 ----
    rec = policy.predict(pil, args.instruction, scale=args.scale)
    mapped = np.asarray(rec["mapped_4d"], dtype=np.float64)

    # ---- 4) 幅度 / 方向合理性校验 ----
    xyz = mapped[:3]
    horiz = float(np.linalg.norm(xyz[:2]))
    toward = target_xy - cube_xy
    cos = (float(np.dot(xyz[:2], toward) /
                 (np.linalg.norm(xyz[:2]) * np.linalg.norm(toward) + 1e-12))
           if horiz > 1e-9 and np.linalg.norm(toward) > 1e-9 else None)
    magnitude_ok = bool(
        np.all(np.abs(xyz) <= args.scale + 1e-6) and
        0.0 <= mapped[3] <= 1.0 and
        horiz <= np.sqrt(2) * args.scale + 1e-6
    )
    sanity = {
        "xyz_abs_max_m": round(float(np.max(np.abs(xyz))), 6),
        "horizontal_step_m": round(horiz, 6),
        "gripper": round(float(mapped[3]), 6),
        "toward_target_cos": None if cos is None else round(cos, 4),
        "direction_label": ("接近目标" if cos is not None and cos > 0.1 else
                            "远离目标" if cos is not None and cos < -0.1 else
                            "横向为主/不可判定"),
        "within_action_space": magnitude_ok,
    }

    # ---- 5) 在 MuJoCo 中执行一步，验证真实位移 ----
    step_result = None
    if not args.no_env_step:
        ee_before = env.data.mocap_pos[0].copy()
        cube_before = env.data.xpos[env._cube_body][:2].copy()
        obs2, reward, term, trunc, info2 = env.step(mapped)
        ee_after = env.data.mocap_pos[0].copy()
        cube_after = env.data.xpos[env._cube_body][:2].copy()
        step_result = {
            "ee_delta": [round(float(v), 6) for v in (ee_after - ee_before)],
            "cube_delta_xy": [round(float(v), 6) for v in (cube_after - cube_before)],
            "dist_before": round(dist0, 6),
            "dist_after": round(float(info2["cube_target_dist"]), 6),
            "dist_change": round(float(info2["cube_target_dist"]) - dist0, 6),
            "reward": round(float(reward), 6),
            "terminated": bool(term), "truncated": bool(trunc),
        }
    env.close()

    result = {
        "config": {
            "model_dir": args.model_dir, "quant": args.quant,
            "unnorm_key": args.unnorm_key, "scale_m": args.scale,
            "instruction": args.instruction, "prompt": rec["prompt"],
            "device": args.device, "seed": args.seed,
        },
        "model": {
            "params_b": round(policy.n_params / 1e9, 3),
            "load_time_s": round(load_s, 2),
            "vram_reserved_gb": policy.memory_gb()["reserved"],
        },
        "image": {"source": image_source, "size": list(pil.size)},
        "state_shapes": state_shapes,
        "env_initial": {
            "target_xy": [round(float(v), 6) for v in target_xy],
            "cube_xy": [round(float(v), 6) for v in cube_xy],
            "dist_to_target": round(dist0, 6),
        },
        "action": {
            "raw_7d": rec["raw_7d"],
            "normalized_7d": rec["normalized_7d"],
            "ignored_rotation": rec["ignored_rotation"],
            "mapped_4d": rec["mapped_4d"],
            "parsed": parse_action(mapped),
            "latency_s": rec["latency_s"],
        },
        "sanity": sanity,
        "mujoco_step": step_result,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[单样本] 模型:", args.model_dir, "量化:", args.quant,
          "参数:", result["model"]["params_b"], "B",
          "显存:", result["model"]["vram_reserved_gb"], "GB")
    print("[单样本] prompt:", rec["prompt"])
    print("[单样本] 原始7维:", np.round(rec["raw_7d"], 5).tolist())
    print("[单样本] 映射4维:", np.round(mapped, 5).tolist())
    print("[单样本] 幅度合法:", magnitude_ok,
          "方向:", sanity["direction_label"],
          "余弦:", sanity["toward_target_cos"])
    if step_result:
        print("[单样本] EE位移:", step_result["ee_delta"],
              "立方体位移:", step_result["cube_delta_xy"],
              "目标距离:", step_result["dist_before"], "->", step_result["dist_after"])
    print("[单样本] 结果已保存:", out_path)


if __name__ == "__main__":
    main()
