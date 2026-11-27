# -*- coding: utf-8 -*-
"""VLA 三类泛化测试（微调 adapter，int4；与 ACT 共用同一批场景）。

用法（vla 环境，仓库根目录）：
  MUJOCO_GL=egl python vla/eval/eval_generalization_vla.py \
    --adapter results/vla_finetune_tuned/best_adapter \
    --scenes results/vla_generalization/scenes.json \
    --out results/vla_generalization/VLA_generalization_raw.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vla" / "utils"))

ADAPTER_PATH = ROOT / "vla" / "utils" / "vla_adapter.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("va_gen", ADAPTER_PATH)
_va = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_va)
build_prompt = _va.build_prompt
preprocess_image = _va.preprocess_image

MUJOCO_SITE = "/opt/miniconda3/envs/mujoco/lib/python3.12/site-packages"
if os.path.isdir(MUJOCO_SITE) and MUJOCO_SITE not in sys.path:
    sys.path.append(MUJOCO_SITE)
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from envs.push_cube_env import PushCubeEnv

MODEL_DIR = "/data/wzh/robot/models/openvla-7b"
SUCCESS_RADIUS = 0.03

# 10 条训练集未见的同义指令（运行时严格校验与训练集零重合）
NOVEL_INSTRUCTIONS = [
    "把小木块挪到前面的红圈里",
    "推动立方体抵达目标标记",
    "将方块推到红色圆形标记处",
    "把蓝色小方块送到红色圆盘上",
    "让立方体滑进桌面的红色目标区",
    "挪动小方块至红色标记圈中央",
    "把积木推向前方的红色靶心",
    "请将蓝色方块移到红色圆标处",
    "推动小立方体落在红色目标圆环内",
    "将桌上的方块滑到红色圈圈里",
]

STANDARD_INSTRUCTION = "push the cube to the red target"


def verify_novel(instructions, train_path):
    """确保测试指令不在训练集指令集合中。"""
    from datasets import load_from_disk
    ds = load_from_disk(str(Path(train_path) / "train"))
    seen = set(str(x).strip().lower() for x in ds["instruction"])
    overlap = [x for x in instructions if x.strip().lower() in seen]
    return seen, overlap


def load_policy(adapter_dir):
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR, quantization_config=bnb, device_map={"": 0},
        low_cpu_mem_usage=True, trust_remote_code=True)
    meta = json.load(open(Path(adapter_dir) / "finetune_meta.json", encoding="utf-8"))
    lo = [float(x) for x in meta["norm_stats"]["min"]]
    hi = [float(x) for x in meta["norm_stats"]["max"]]
    key = meta["unnorm_key"]
    model.config.norm_stats[key] = {"action": {
        "q01": lo[:3] + [0, 0, 0] + [lo[3]],
        "q99": hi[:3] + [0, 0, 0] + [hi[3]],
        "mask": [True, True, True, False, False, False, True],
    }}
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    vram = torch.cuda.memory_reserved() / 1e9
    print(f"[model] adapter={adapter_dir} unnorm_key={key} vram={vram:.2f}GB", flush=True)
    return processor, model, key


def predict(processor, model, key, rgb, instruction):
    pil, _ = preprocess_image(rgb)
    inputs = processor(build_prompt(instruction), pil).to("cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        raw = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
        torch.cuda.synchronize()
    r = np.asarray(raw, dtype=np.float64).reshape(-1)[:7]
    # 微调口径：raw 前三维已是物理位移（direct_phys）
    return [float(r[0]), float(r[1]), float(r[2]), float((r[6] + 1.0) / 2.0)]


def run_episode(env, processor, model, key, scene, instruction, max_steps):
    obs, info = env.reset(options={"scene": scene})
    img = np.asarray(obs["image"])
    if img.ndim != 3 or img.shape[-1] != 3 or float(img.max()) <= float(img.min()):
        raise RuntimeError("invalid image; need MUJOCO_GL=egl")
    dist0 = float(info["cube_target_dist"])
    dists = [dist0]
    success = False
    t0 = time.perf_counter()
    for t in range(max_steps):
        action = predict(processor, model, key, obs["image"], instruction)
        obs, reward, term, trunc, info = env.step(np.asarray(action))
        dists.append(float(info["cube_target_dist"]))
        success = bool(info.get("success", False))
        if term or trunc:
            break
    return {
        "instruction": instruction,
        "kind": scene["kind"], "scene_id": int(scene["scene_id"]),
        "cube_xy": scene["cube_xy"], "target_xy": scene["target_xy"],
        "dist_start": round(dist0, 6), "dist_end": round(dists[-1], 6),
        "dist_min": round(min(dists), 6),
        "dist_improvement": round(dist0 - dists[-1], 6),
        "success": success, "steps": t + 1,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }


def summarize(records):
    n = len(records)
    succ = [r for r in records if r["success"]]
    return {
        "n_episodes": n,
        "success_count": len(succ),
        "success_rate": round(len(succ) / n, 4) if n else 0.0,
        "mean_dist_improvement": round(float(np.mean([r["dist_improvement"] for r in records])), 6) if n else None,
        "mean_dist_end": round(float(np.mean([r["dist_end"] for r in records])), 6) if n else None,
        "mean_steps": round(float(np.mean([r["steps"] for r in records])), 2) if n else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--train-dataset", default="/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=50)
    args = ap.parse_args()

    payload = json.load(open(args.scenes, encoding="utf-8"))
    scenes = payload["scenes"]
    seen, overlap = verify_novel(NOVEL_INSTRUCTIONS, args.train_dataset)
    print(f"[verify] train instructions={len(seen)} novel={len(NOVEL_INSTRUCTIONS)} overlap={len(overlap)}", flush=True)
    if overlap:
        raise RuntimeError(f"novel instructions overlap with training set: {overlap}")

    processor, model, key = load_policy(args.adapter)
    env = PushCubeEnv(scene_override=True, obs_mode="both", rgb_preset="demo",
                      render_camera="global_cam", render_mode="rgb_array",
                      max_episode_steps=args.max_steps)

    by_kind = {}
    for s in scenes:
        by_kind.setdefault(s["kind"], []).append(s)
    records = {"instruction": [], "start_pos": [], "target_pos": []}
    t_start = time.perf_counter()

    # 1) 指令泛化：10 条新指令 × 3 重复场景
    for instr in NOVEL_INSTRUCTIONS:
        for sc in by_kind["instruction"]:
            rec = run_episode(env, processor, model, key, sc, instr, args.max_steps)
            records["instruction"].append(rec)
            print(f"[INS] {instr[:22]:<22} scene={rec['scene_id']} succ={rec['success']} "
                  f"dist {rec['dist_start']:.3f}->{rec['dist_end']:.3f}", flush=True)

    # 2) 起点泛化：20 场景，标准指令
    for sc in by_kind["start_pos"]:
        rec = run_episode(env, processor, model, key, sc, STANDARD_INSTRUCTION, args.max_steps)
        records["start_pos"].append(rec)
        print(f"[START] scene={rec['scene_id']} succ={rec['success']} "
              f"dist {rec['dist_start']:.3f}->{rec['dist_end']:.3f}", flush=True)

    # 3) 目标泛化：20 场景，标准指令
    for sc in by_kind["target_pos"]:
        rec = run_episode(env, processor, model, key, sc, STANDARD_INSTRUCTION, args.max_steps)
        records["target_pos"].append(rec)
        print(f"[TARGET] {sc.get('direction','?')} scene={rec['scene_id']} succ={rec['success']} "
              f"dist {rec['dist_start']:.3f}->{rec['dist_end']:.3f}", flush=True)

    env.close()
    summary = {k: summarize(v) for k, v in records.items()}
    summary["overall"] = summarize([r for v in records.values() for r in v])
    out = {
        "algorithm": "vla_openvla_lora",
        "adapter": args.adapter,
        "unnorm_key": key,
        "observation": "224x224 RGB + language instruction",
        "protocol": {
            "scene_file": args.scenes, "scene_master_seed": payload.get("master_seed"),
            "max_steps": args.max_steps, "success_radius_m": SUCCESS_RADIUS,
            "instruction_mode": "direct_phys (finetuned)",
            "novel_instructions": NOVEL_INSTRUCTIONS,
            "novel_overlap_with_train": overlap,
            "train_instruction_count": len(seen),
        },
        "summary": summary,
        "per_test": records,
        "total_wall_s": round(time.perf_counter() - t_start, 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 68)
    for k in ("instruction", "start_pos", "target_pos", "overall"):
        s = summary[k]
        print(f"[VLA] {k}: {s['success_count']}/{s['n_episodes']} rate={s['success_rate']} "
              f"dist_imp={s['mean_dist_improvement']}")
    print("[VLA] raw ->", args.out, flush=True)
    print("VLA_GENERALIZATION_DONE")


if __name__ == "__main__":
    main()
