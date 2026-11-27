# -*- coding: utf-8 -*-
"""跨任务（推到左边/右边）闭环评估：VLA 基线 / VLA few-shot / ACT 共用场景。

用法（vla 环境，注入 mujoco site-packages）：
  python vla/eval/eval_cross_task.py --algo vla \
    --adapter <adapter_dir> --scenes results/vla_cross_task/scenes.json \
    --out <raw.json> [--episodes-per-task 8]
  python vla/eval/eval_cross_task.py --algo act \
    --checkpoint /data/wzh/robot/outputs/act_chunk40/checkpoints/004000 \
    --scenes results/vla_cross_task/scenes.json --out <raw.json>
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
_spec = importlib.util.spec_from_file_location("va_ct", ADAPTER_PATH)
_va = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_va)
build_prompt = _va.build_prompt
preprocess_image = _va.preprocess_image

MUJOCO_SITE = "/opt/miniconda3/envs/mujoco/lib/python3.12/site-packages"
if os.path.isdir(MUJOCO_SITE) and MUJOCO_SITE not in sys.path:
    sys.path.append(MUJOCO_SITE)
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")

import torch
from envs.push_cube_env import PushCubeEnv

MODEL_DIR = "/data/wzh/robot/models/openvla-7b"
SUCCESS_RADIUS = 0.03
INSTR = {"left": "把方块推到左边", "right": "把方块推到右边"}


def load_vla(adapter_dir):
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
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


def predict_vla(processor, model, key, rgb, instruction):
    pil, _ = preprocess_image(rgb)
    inputs = processor(build_prompt(instruction), pil).to("cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        raw = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
        torch.cuda.synchronize()
    r = np.asarray(raw, dtype=np.float64).reshape(-1)[:7]
    # direct_phys: 微调模式 raw 前 3 维即物理位移；gripper 置 0
    return np.array([r[0], r[1], r[2], 0.0], dtype=np.float32)


def load_act(ckpt):
    import os
    for pp in ("/data/wzh/robot/embodied-policy-zoo",
               "/data/wzh/robot/embodied-policy-zoo/scripts",
               "/data/wzh/robot/mujoco-studio-lab",
               "/data/wzh/robot/mujoco-studio-lab/envs"):
        if pp not in sys.path:
            sys.path.insert(0, pp)
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.envs import make_env_config
    import lerobot_push_cube_env  # noqa: F401
    pretrained = os.path.join(ckpt, "pretrained_model")
    if not os.path.isdir(pretrained):
        pretrained = ckpt
    cfg = ACTConfig.from_pretrained(pretrained)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = make_policy(cfg=cfg, env_cfg=make_env_config("push_cube"),
                         pretrained_path=pretrained)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=pretrained)
    device = next(policy.parameters()).device
    return policy, pre, post, device


def run_episode(env, algo_handle, scene, instruction, max_steps):
    obs, info = env.reset(options={"scene": scene})
    dist0 = float(info["cube_target_dist"])
    dmin = dist0
    for t in range(max_steps):
        if algo_handle["algo"] == "vla":
            action = predict_vla(algo_handle["processor"], algo_handle["model"],
                                 algo_handle["key"], obs["image"], instruction)
        else:
            dev = algo_handle["device"]
            img = np.transpose(obs["image"], (2, 0, 1)).astype(np.float32) / 255.0
            canon = {
                "observation.images.main": torch.from_numpy(
                    np.ascontiguousarray(img)[None]).to(dev),
                "observation.state": torch.from_numpy(
                    np.ascontiguousarray(obs["state"]).astype(np.float32)[None]).to(dev),
            }
            with torch.inference_mode():
                processed = algo_handle["pre"](canon)
                act = algo_handle["policy"].select_action(processed)
                act = algo_handle["post"](act)
            if isinstance(act, dict):
                act = act["action"]
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()
            action = np.clip(np.asarray(act).reshape(-1)[:4],
                             env.action_space.low, env.action_space.high)
        obs, r, term, trunc, info = env.step(action)
        d = float(info["cube_target_dist"])
        dmin = min(dmin, d)
        if info.get("success"):
            break
    dist_end = float(info["cube_target_dist"])
    return {
        "task": scene["task"], "scene_id": int(scene["scene_id"]),
        "instruction": instruction,
        "target_xy": scene["target_xy"], "cube_xy": scene["cube_xy"],
        "dist_start": round(dist0, 4), "dist_end": round(dist_end, 4),
        "dist_min": round(dmin, 4),
        "success": bool(dist_end < SUCCESS_RADIUS),
        "steps": int(info["steps"]),
    }


def summarize(records):
    out = {}
    for task in ("left", "right"):
        rs = [r for r in records if r["task"] == task]
        n = len(rs); succ = sum(r["success"] for r in rs)
        out[task] = {
            "n_episodes": n, "success_count": succ,
            "success_rate": round(succ / n, 4) if n else 0.0,
            "mean_dist_improvement": round(float(np.mean(
                [r["dist_start"] - r["dist_end"] for r in rs])) if n else 0.0, 4),
        }
    all_n = len(records); all_succ = sum(r["success"] for r in records)
    out["overall"] = {
        "n_episodes": all_n, "success_count": all_succ,
        "success_rate": round(all_succ / all_n, 4) if all_n else 0.0,
        "mean_dist_improvement": round(float(np.mean(
            [r["dist_start"] - r["dist_end"] for r in records])) if all_n else 0.0, 4),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=("vla", "act"), required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--episodes-per-task", type=int, default=0, help="0=用场景中全部")
    args = ap.parse_args()

    payload = json.load(open(args.scenes, encoding="utf-8"))
    scenes = payload["scenes"]
    if args.episodes_per_task > 0:
        by = {}
        for s in scenes:
            by.setdefault(s["task"], []).append(s)
        scenes = [s for t in ("left", "right")
                  for s in by[t][:args.episodes_per_task]]

    algo_handle = {}
    if args.algo == "vla":
        p, m, k = load_vla(args.adapter)
        algo_handle = {"algo": "vla", "processor": p, "model": m, "key": k}
    else:
        p, pre, post, dev = load_act(args.checkpoint)
        algo_handle = {"algo": "act", "policy": p, "pre": pre, "post": post, "device": dev}

    env = PushCubeEnv(scene_override=True, obs_mode="both", rgb_preset="demo",
                      max_episode_steps=args.max_steps)
    records = []
    for i, sc in enumerate(scenes):
        rec = run_episode(env, algo_handle, sc, INSTR[sc["task"]], args.max_steps)
        records.append(rec)
        print(f"[{args.algo.upper()}-{sc['task']}] scene={rec['scene_id']} "
              f"succ={rec['success']} dist {rec['dist_start']:.3f}->{rec['dist_end']:.3f} "
              f"min={rec['dist_min']:.3f}", flush=True)
    env.close()

    summary = summarize(records)
    out = {
        "algorithm": args.algo,
        "adapter" if args.algo == "vla" else "checkpoint": (
            args.adapter if args.algo == "vla" else args.checkpoint),
        "scenes_file": args.scenes,
        "max_steps": args.max_steps,
        "success_radius_m": SUCCESS_RADIUS,
        "instructions": INSTR,
        "note": ("VLA: direct_phys 微调动作映射" if args.algo == "vla" else
                 "ACT: 视觉+13维状态输入，无语言条件；指令仅为固定任务语义标记"),
        "summary": summary,
        "episodes": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for k in ("left", "right", "overall"):
        s = summary[k]
        print(f"[{args.algo.upper()}] {k}: {s['success_count']}/{s['n_episodes']} "
              f"rate={s['success_rate']} dist_imp={s['mean_dist_improvement']}", flush=True)
    print("CROSS_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
