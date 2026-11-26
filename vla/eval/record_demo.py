#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""录制微调后模型执行标准指令的完整闭环演示视频。

标准指令："push the cube to the red target"（把方块推到红色目标处）。
逐帧保存 MuJoCo 渲染画面，执行完整回合后用 ffmpeg 合成 mp4。
复用 eval.py 的模型加载/动作映射口径（微调 adapter + pushcube_vla）。

**必须设置 MUJOCO_GL=egl**，否则渲染为全零占位。

用法（vla 环境，仓库根目录）：
    export MUJOCO_GL=egl
    python vla/eval/record_demo.py --adapter results/vla_finetune_tuned/best_adapter \
        --instruction "push the cube to the red target" --max-steps 50 \
        --out-dir /data/wzh/robot/datasets/vla_pushcube_finetune/demo_video
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

# 复用 eval.py 的加载与推理函数（同目录）
import importlib.util
_EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval.py")
_spec = importlib.util.spec_from_file_location("eval_mod", _EVAL_PATH)
eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_mod)


def parse_args():
    ap = argparse.ArgumentParser(description="录制微调模型闭环演示视频")
    ap.add_argument("--model-dir", default="/data/wzh/robot/models/openvla-7b")
    ap.add_argument("--adapter", default="results/vla_finetune_tuned/best_adapter")
    ap.add_argument("--instruction", default="push the cube to the red target")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--scale", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    root = eval_mod.ROOT
    adapter = args.adapter if os.path.isabs(args.adapter) else os.path.join(root, args.adapter)
    out_dir = args.out_dir or os.path.join(
        "/data/wzh/robot/datasets/vla_pushcube_finetune", "demo_video")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 复用 load_policy 所需的命名空间（用 SimpleNamespace 避免类体作用域问题）
    from types import SimpleNamespace
    load_args = SimpleNamespace(
        mode="finetuned", model_dir=args.model_dir, adapter=adapter, device="cuda:0")

    print(f"[demo] 加载微调 adapter={adapter}", flush=True)
    processor, model, unnorm_key = eval_mod.load_policy(load_args)
    env = eval_mod.make_env(args.max_steps)

    obs, info = env.reset(seed=args.seed)
    img = np.asarray(obs["image"])
    if img.ndim != 3 or img.shape[-1] != 3 or float(img.max()) <= float(img.min()):
        raise RuntimeError("观测图像无效，请先 export MUJOCO_GL=egl")

    dist0 = float(info["cube_target_dist"])
    dists = [dist0]
    records = []
    saved = 0

    def save_frame(step_idx, dist, success):
        frame = env.render()  # RGB array
        pil = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(pil)
        # 左上角叠加信息条（不遮挡操作区）
        label = f"step {step_idx:02d} | dist {dist:.3f} | {'SUCCESS' if success else 'running'}"
        draw.rectangle([0, 0, 460, 22], fill=(0, 0, 0))
        draw.text((6, 5), label, fill=(255, 255, 255))
        pil.save(os.path.join(frames_dir, f"frame_{step_idx:04d}.png"))

    save_frame(0, dist0, False)
    saved += 1
    success = False
    t0 = time.perf_counter()
    for step in range(args.max_steps):
        rec = eval_mod.predict_action(processor, model, unnorm_key,
                                      obs["image"], args.instruction, args.scale,
                                      direct_phys=True)
        mapped = np.asarray(rec["mapped_4d"], dtype=np.float64)
        obs, reward, term, trunc, info = env.step(mapped)
        dists.append(float(info["cube_target_dist"]))
        success = bool(info.get("success", False))
        records.append({
            "step": step + 1,
            "mapped_4d": [round(float(v), 6) for v in mapped],
            "dist_to_target": round(dists[-1], 6),
            "success": success,
        })
        save_frame(step + 1, dists[-1], success)
        saved += 1
        if term or trunc:
            break
    env.close()
    wall = time.perf_counter() - t0

    # 合成 mp4（ffmpeg 从图像序列生成，稳定可靠）
    video_path = os.path.join(out_dir, "demo.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(args.fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        video_path,
    ]
    subprocess.run(cmd, check=True)
    size_mb = round(os.path.getsize(video_path) / 1e6, 2)

    meta = {
        "instruction": args.instruction,
        "instruction_zh": "把方块推到红色目标处",
        "model": "OpenVLA-7B int4 + LoRA adapter (best by eval_loss)",
        "adapter": adapter,
        "unnorm_key": unnorm_key,
        "seed": args.seed,
        "dist_start": round(dist0, 6),
        "dist_end": round(dists[-1], 6),
        "dist_min": round(min(dists), 6),
        "success": success,
        "steps": len(records),
        "frames": saved,
        "fps": args.fps,
        "duration_s": round(saved / args.fps, 2),
        "wall_time_s": round(wall, 2),
        "video": video_path,
        "video_size_mb": size_mb,
        "steps_detail": records,
    }
    with open(os.path.join(out_dir, "demo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[demo] 完成：{saved} 帧 -> {video_path} ({size_mb} MB) "
          f"dist {dist0:.4f}→{dists[-1]:.4f} success={success} "
          f"steps={len(records)} 耗时={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
