#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量闭环评估：微调后模型 vs 官方零样本模型的 PushCube 任务成功率。

在测试集指令上跑完整回合，统计每条指令/总体成功率、距离改善、步数、延迟。
两种模式共用同一套环境与映射口径，只有模型权重与 unnorm_key 不同：

- 微调模式（默认）：底座 int4 + PEFT adapter（验证 loss 最低的 best_adapter），
  unnorm_key=pushcube_vla（训练时注册的自采统计量）；
- 零样本模式（--zero-shot）：纯底座 int4，unnorm_key=bridge_orig。

在 vla 进程内注入 mujoco 环境依赖，模型只加载一次、环境驻留。
**必须设置 MUJOCO_GL=egl**，否则 obs["image"] 是全零占位，评估无效。

用法（vla 环境，仓库根目录）：
    export MUJOCO_GL=egl
    python vla/eval/eval.py --mode finetuned \
        --adapter results/vla_finetune_tuned/best_adapter \
        --instructions "push the cube to the red target" ... \
        --episodes 5 --max-steps 40 --out results/vla_eval/finetuned.json
"""

from __future__ import annotations

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

# 按文件路径加载适配器，避免同名 utils 包遮蔽
_ADAPTER_PATH = os.path.join(ROOT, "vla", "utils", "vla_adapter.py")
_spec = importlib.util.spec_from_file_location("vla_adapter_eval", _ADAPTER_PATH)
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)
build_prompt = _adapter.build_prompt
preprocess_image = _adapter.preprocess_image
map_action = _adapter.map_action

# 注入 mujoco 环境依赖（gymnasium/mujoco）
MUJOCO_SITE = "/opt/miniconda3/envs/mujoco/lib/python3.12/site-packages"
if os.path.isdir(MUJOCO_SITE) and MUJOCO_SITE not in sys.path:
    sys.path.append(MUJOCO_SITE)
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")

from envs.push_cube_env import PushCubeEnv  # noqa: E402

from transformers import (  # noqa: E402
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel  # noqa: E402

# 测试集中的代表性的标准指令（非左右变体，标准红盘目标）
DEFAULT_INSTRUCTIONS = [
    "push the cube to the red target",
    "push the cube into the red marked area",
    "move block to red target",
    "slide the cube ahead to the red goal",
    "push the block onto the red disc",
]


def parse_args():
    ap = argparse.ArgumentParser(description="OpenVLA 批量闭环评估")
    ap.add_argument("--mode", choices=("finetuned", "zero-shot"), default="finetuned")
    ap.add_argument("--model-dir", default="/data/wzh/robot/models/openvla-7b")
    ap.add_argument("--adapter", default=None,
                    help="微调 adapter 目录（finetuned 模式必填）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--instructions", nargs="+", default=DEFAULT_INSTRUCTIONS)
    ap.add_argument("--episodes", type=int, default=3,
                    help="每条指令的回合数（不同 seed）")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--scale", type=float, default=0.02)
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    return ap.parse_args()


def register_norm_stats(model, adapter_dir=None):
    """注册推理用的 unnorm_key。

    微调模式：从 adapter 的 finetune_meta.json 读自采 min/max，注册 pushcube_vla，
    与训练时 register 的统计量逐位一致（q01/q99 各 7 维，旋转维置 0）。
    零样本模式：直接用底座自带的 bridge_orig。
    """
    if adapter_dir:
        meta_path = os.path.join(adapter_dir, "finetune_meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        lo = [float(x) for x in meta["norm_stats"]["min"]]
        hi = [float(x) for x in meta["norm_stats"]["max"]]
        key = meta["unnorm_key"]
        if not hasattr(model.config, "norm_stats") or model.config.norm_stats is None:
            model.config.norm_stats = {}
        # 注意：predict_action 读 norm_stats[key]["action"]["q01"]，
        # map_action 亦读 ["action"]，必须套一层 action 键。
        model.config.norm_stats[key] = {
            "action": {
                "q01": lo[:3] + [0, 0, 0] + [lo[3]],
                "q99": hi[:3] + [0, 0, 0] + [hi[3]],
                "mask": [True, True, True, False, False, False, True],
            }
        }
        print(f"[norm] 注册 {key} min={lo} max={hi}", flush=True)
        return key
    return "bridge_orig"


def load_policy(args):
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_dir,
        quantization_config=bnb,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    unnorm_key = register_norm_stats(model, args.adapter if args.mode == "finetuned" else None)
    if args.mode == "finetuned":
        if not args.adapter or not os.path.isdir(args.adapter):
            raise FileNotFoundError(f"finetuned 模式需要 --adapter 目录: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] PEFT adapter 已加载: {args.adapter} 可训练参数={trainable:,}", flush=True)
    model.eval()
    vram = torch.cuda.memory_reserved() / 1e9
    print(f"[model] 模式={args.mode} unnorm_key={unnorm_key} 显存保留={vram:.2f}GB "
          f"加载耗时={time.perf_counter()-t0:.1f}s", flush=True)
    return processor, model, unnorm_key


def make_env(max_steps):
    return PushCubeEnv(
        obs_mode="both", rgb_preset="demo", render_camera="global_cam",
        render_mode="rgb_array", randomize_start=False,
        max_episode_steps=max_steps,
    )


def predict_action(processor, model, unnorm_key, rgb, instruction, scale,
                   direct_phys=False):
    """推理一步。

    direct_phys=True（微调模式）：predict_action 用 pushcube_vla 统计量反归一化，
    返回的 raw 前 3 维**已经是 PushCube 物理位移（米）**，与训练采集口径一致，
    直接作为位置增量；不可再经 unnorm_to_unit × scale 二次映射，否则 unit=0
    （训练中对应 dx≈0.01m）会被错映射成"不动"，策略完全失效。

    direct_phys=False（零样本模式）：底座用 bridge_orig 统计量，raw 是桥数据集
    动作量纲，沿用既有 map_action 缩放到本环境量程。
    """
    pil, _ = preprocess_image(rgb)
    prompt = build_prompt(instruction)
    inputs = processor(prompt, pil).to("cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        raw = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        torch.cuda.synchronize()
    if direct_phys:
        r = np.asarray(raw, dtype=np.float64).reshape(-1)[:7]
        return {
            "mapped_4d": [float(r[0]), float(r[1]), float(r[2]),
                          float((r[6] + 1.0) / 2.0)],
            "raw_7d": [float(v) for v in r],
            "normalized_7d": None,
            "scale_m": None,
            "mapping": "direct_phys",
        }
    return map_action(raw, model.config.norm_stats, unnorm_key, scale=scale)


def run_episode(env, processor, model, unnorm_key, instruction, args, seed):
    """跑一个完整回合，返回汇总指标。"""
    obs, info = env.reset(seed=seed)
    # 拦截无效图像（未设 MUJOCO_GL=egl 时 obs['image'] 是全零占位）
    img = np.asarray(obs["image"])
    if img.ndim != 3 or img.shape[-1] != 3 or float(img.max()) <= float(img.min()):
        raise RuntimeError(
            f"观测图像无效 shape={img.shape} max={img.max()} min={img.min()}；"
            "请先 export MUJOCO_GL=egl 再运行评估")
    dist0 = float(info["cube_target_dist"])
    dists = [dist0]
    latencies = []
    success = False
    steps = 0
    t0 = time.perf_counter()
    for step in range(args.max_steps):
        rec = predict_action(processor, model, unnorm_key, obs["image"],
                             instruction, args.scale,
                             direct_phys=(args.mode == "finetuned"))
        mapped = np.asarray(rec["mapped_4d"], dtype=np.float64)
        obs, reward, term, trunc, info = env.step(mapped)
        dists.append(float(info["cube_target_dist"]))
        success = bool(info.get("success", False))
        steps = step + 1
        if term or trunc:
            break
    wall = time.perf_counter() - t0
    return {
        "instruction": instruction,
        "seed": seed,
        "dist_start": round(dist0, 6),
        "dist_end": round(dists[-1], 6),
        "dist_min": round(min(dists), 6),
        "dist_improvement": round(dist0 - dists[-1], 6),
        "success": success,
        "steps": steps,
        "wall_time_s": round(wall, 2),
    }


def main():
    args = parse_args()
    print(f"[eval] 模式={args.mode} 指令数={len(args.instructions)} "
          f"每指令回合={args.episodes} 最大步数={args.max_steps} "
          f"量程=±{args.scale}m", flush=True)
    processor, model, unnorm_key = load_policy(args)
    env = make_env(args.max_steps)
    results = []
    t_start = time.perf_counter()
    for i, instr in enumerate(args.instructions):
        for j in range(args.episodes):
            seed = args.seed_base + i * 100 + j
            ep = run_episode(env, processor, model, unnorm_key, instr, args, seed)
            results.append(ep)
            print(f"[EP] {instr[:38]:<38} seed={seed} "
                  f"dist {ep['dist_start']:.4f}→{ep['dist_end']:.4f} "
                  f"(min {ep['dist_min']:.4f}) success={ep['success']} "
                  f"steps={ep['steps']} {ep['wall_time_s']:.1f}s", flush=True)
    env.close()

    # 按指令聚合
    per_instr = {}
    for instr in args.instructions:
        rows = [r for r in results if r["instruction"] == instr]
        per_instr[instr] = {
            "episodes": len(rows),
            "successes": sum(r["success"] for r in rows),
            "success_rate": round(sum(r["success"] for r in rows) / max(len(rows), 1), 4),
            "mean_dist_improvement": round(
                float(np.mean([r["dist_improvement"] for r in rows])), 6),
        }
    n = len(results)
    n_succ = sum(r["success"] for r in results)
    summary = {
        "mode": args.mode,
        "unnorm_key": unnorm_key,
        "adapter": args.adapter if args.mode == "finetuned" else None,
        "config": {
            "max_steps": args.max_steps, "scale_m": args.scale,
            "episodes_per_instruction": args.episodes, "seed_base": args.seed_base,
        },
        "episodes": results,
        "per_instruction": per_instr,
        "total_episodes": n,
        "total_successes": n_succ,
        "success_rate": round(n_succ / max(n, 1), 4),
        "mean_dist_improvement": round(
            float(np.mean([r["dist_improvement"] for r in results])), 6),
        "total_wall_s": round(time.perf_counter() - t_start, 2),
    }
    out = args.out or os.path.join(ROOT, "results", "vla_eval", f"{args.mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[eval] 完成 模式={args.mode} 成功率={summary['success_rate']} "
          f"({n_succ}/{n}) 平均距离改善={summary['mean_dist_improvement']}m "
          f"总耗时={summary['total_wall_s']}s", flush=True)
    print(f"[eval] 结果 -> {out}", flush=True)


if __name__ == "__main__":
    main()
