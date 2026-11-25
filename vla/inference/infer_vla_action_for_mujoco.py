#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 OpenVLA 对自定义 MuJoCo 观测做单步推理，并映射到 PushCube 动作空间（vla 环境）。

映射规则：丢弃 OpenVLA 输出的 3 维旋转增量（drx/dry/drz），只取 dx/dy/dz；
先用数据集 q01/q99 把反归一化的物理增量还原到 [-1,1]，再线性缩放到 ±scale；
夹爪维按 (g+1)/2 映射到 [0,1]（PushCube 中该位保留、不参与动力学）。
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args():
    ap = argparse.ArgumentParser(description="OpenVLA → PushCube 单步动作映射")
    ap.add_argument("--model-dir", default="/data/wzh/robot/models/openvla-7b")
    ap.add_argument("--image", default=None)
    ap.add_argument("--instruction", default="push the cube to the red target")
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--scale", type=float, default=0.02, help="每步位移上限（米）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    image_path = args.image or os.path.join(root, "results", "vla_single_step", "observation_224.png")
    out_path = args.out or os.path.join(root, "results", "vla_single_step", "vla_action.json")

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device)
    model.eval()
    image = Image.open(image_path).convert("RGB").resize((224, 224), Image.LANCZOS)
    prompt = f"In: What action should the robot take to {args.instruction}?\nOut:"
    inputs = processor(prompt, image).to(args.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    raw = np.asarray(action).reshape(-1).astype(float)[:7]
    stats = model.config.norm_stats[args.unnorm_key]["action"]
    q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
    norm = np.clip(2.0 * (raw - q01) / np.maximum(q99 - q01, 1e-8) - 1.0, -1.0, 1.0)
    xyz = (norm[:3] * args.scale).astype(float)
    gripper = float((norm[6] + 1.0) / 2.0)
    mapped = [float(v) for v in xyz] + [gripper]

    rec = {
        "image": image_path, "instruction": args.instruction, "prompt": prompt,
        "unnorm_key": args.unnorm_key, "scale_m": args.scale,
        "raw_7d": raw.tolist(), "raw_names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        "normalized_7d": norm.tolist(),
        "ignored_rotation": norm[3:6].tolist(),
        "mapped_4d": mapped,
        "mapped_names": ["dx", "dy", "dz", "gripper"],
        "latency_s": round(latency, 4),
        "vram_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("[VLA2MUJOCO] prompt:", prompt)
    print("[VLA2MUJOCO] 原始7维:", np.round(raw, 5).tolist())
    print("[VLA2MUJOCO] 归一化7维:", np.round(norm, 5).tolist())
    print("[VLA2MUJOCO] 忽略旋转:", np.round(norm[3:6], 5).tolist())
    print("[VLA2MUJOCO] 映射4维:", np.round(mapped, 5).tolist())
    print("[VLA2MUJOCO] 耗时:", rec["latency_s"], "s，结果:", out_path)


if __name__ == "__main__":
    main()
