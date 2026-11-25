#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenVLA 官方示例单帧推理与性能基线（vla 环境）。

示例图来自官方项目页 openvla.github.io 的 teaser 机器人观测画面。
输出 7 维末端增量动作，并记录耗时 / 显存基线。
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
    ap = argparse.ArgumentParser(description="OpenVLA 官方示例单帧推理")
    ap.add_argument("--model-dir", default="/data/wzh/robot/models/openvla-7b")
    ap.add_argument("--image", default=None, help="默认取仓库 results 下的官方示例图")
    ap.add_argument("--instruction", default="put the yellow corn on the plate")
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=5, help="计时重复次数")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    image_path = args.image or os.path.join(root, "results", "openvla_official_demo.jpg")
    out_path = args.out or os.path.join(root, "results", "openvla_official_demo_inference.json")

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
        model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)  # 预热
        torch.cuda.synchronize()
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    raw = np.asarray(action).reshape(-1).astype(float)[:7]
    names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    stats = model.config.norm_stats[args.unnorm_key]["action"]
    q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
    norm = np.clip(2.0 * (raw - q01) / np.maximum(q99 - q01, 1e-8) - 1.0, -1.0, 1.0)
    mem = {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
    }
    rec = {
        "image": image_path, "instruction": args.instruction,
        "prompt": prompt, "unnorm_key": args.unnorm_key,
        "action_raw": {n: round(float(v), 5) for n, v in zip(names, raw)},
        "action_normalized": {n: round(float(v), 5) for n, v in zip(names, norm)},
        "latency_s": {
            "mean": round(float(np.mean(times)), 4),
            "std": round(float(np.std(times)), 4),
            "min": round(float(np.min(times)), 4),
            "max": round(float(np.max(times)), 4),
            "hz": round(1.0 / float(np.mean(times)), 2),
        },
        "vram": mem,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("[OFFICIAL] prompt:", prompt)
    print("[OFFICIAL] 7维动作:", rec["action_raw"])
    print("[OFFICIAL] 归一化:", rec["action_normalized"])
    print("[OFFICIAL] 耗时:", rec["latency_s"], "显存:", mem)
    print("[OFFICIAL] 结果已保存:", out_path)


if __name__ == "__main__":
    main()
