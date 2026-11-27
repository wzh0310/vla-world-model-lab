# -*- coding: utf-8 -*-
"""ACT 三类泛化测试（lerobot 环境；与 VLA 共用同一批场景）。

ACT 无语言条件：指令泛化一类对 ACT 而言是「固定场景重复 3 次×10 条指令」，
不代表语言泛化，仅用于给出同协议下的对照基线，报告中如实标注。

用法（lerobot 环境）：
  MUJOCO_GL=egl python vla/eval/eval_generalization_act.py \
    --checkpoint /data/wzh/robot/outputs/act_chunk40/checkpoints/004000 \
    --scenes results/vla_generalization/scenes.json \
    --out results/vla_generalization/ACT_generalization_raw.json
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
for p in ("/data/wzh/robot/embodied-policy-zoo",
          "/data/wzh/robot/embodied-policy-zoo/scripts",
          "/data/wzh/robot/mujoco-studio-lab",
          "/data/wzh/robot/mujoco-studio-lab/envs"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from pushcube_envs.push_cube_env import PushCubeEnv
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.envs import make_env_config
import lerobot_push_cube_env  # noqa: F401

SUCCESS_RADIUS = 0.03
SEED_BASE = 900000
NOVEL_INSTRUCTIONS = [
    "把小木块挪到前面的红圈里", "推动立方体抵达目标标记",
    "将方块推到红色圆形标记处", "把蓝色小方块送到红色圆盘上",
    "让立方体滑进桌面的红色目标区", "挪动小方块至红色标记圈中央",
    "把积木推向前方的红色靶心", "请将蓝色方块移到红色圆标处",
    "推动小立方体落在红色目标圆环内", "将桌上的方块滑到红色圈圈里",
]
STANDARD_INSTRUCTION = "push the cube to the red target"


def load_policy(ckpt):
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
    print(f"[ACT] ckpt={ckpt} device={device}", flush=True)
    return policy, pre, post, device


def run_episode(env, policy, pre, post, device, scene, max_steps, seed):
    obs, info = env.reset(options={"scene": scene})
    torch.manual_seed(seed)
    ep_reward = 0.0
    for t in range(max_steps):
        img = np.transpose(obs["image"], (2, 0, 1)).astype(np.float32) / 255.0
        canon = {
            "observation.images.main": torch.from_numpy(
                np.ascontiguousarray(img)[None]).to(device),
            "observation.state": torch.from_numpy(
                np.ascontiguousarray(obs["state"]).astype(np.float32)[None]).to(device),
        }
        with torch.inference_mode():
            processed = pre(canon)
            action = policy.select_action(processed)
            action = post(action)
        if isinstance(action, dict):
            action = action["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        raw = np.clip(np.asarray(action).reshape(-1)[:4],
                      env.action_space.low, env.action_space.high)
        obs, reward, term, trunc, info = env.step(raw)
        ep_reward += float(reward)
        if term or trunc:
            break
    cube = env.data.xpos[env._cube_body]
    dist = float(np.linalg.norm(cube[:2] - env.target_xy))
    return {
        "scene_id": int(scene["scene_id"]), "kind": scene["kind"],
        "cube_xy": scene["cube_xy"], "target_xy": scene["target_xy"],
        "dist_start": round(float(np.linalg.norm(
            np.asarray(scene["cube_xy"]) - np.asarray(scene["target_xy"]))), 6),
        "dist_end": round(dist, 6),
        "dist_improvement": round(float(np.linalg.norm(
            np.asarray(scene["cube_xy"]) - np.asarray(scene["target_xy"]))) - dist, 6),
        "success": bool(dist < SUCCESS_RADIUS), "steps": t + 1,
        "reward": round(ep_reward, 4),
    }


def summarize(records):
    n = len(records)
    succ = [r for r in records if r["success"]]
    return {
        "n_episodes": n, "success_count": len(succ),
        "success_rate": round(len(succ) / n, 4) if n else 0.0,
        "mean_dist_improvement": round(float(np.mean([r["dist_improvement"] for r in records])), 6) if n else None,
        "mean_dist_end": round(float(np.mean([r["dist_end"] for r in records])), 6) if n else None,
        "mean_steps": round(float(np.mean([r["steps"] for r in records])), 2) if n else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    payload = json.load(open(args.scenes, encoding="utf-8"))
    scenes = payload["scenes"]
    if args.limit:
        scenes = scenes[:args.limit]
    policy, pre, post, device = load_policy(args.checkpoint)
    env = PushCubeEnv(obs_mode="both", scene_override=True,
                      max_episode_steps=args.max_steps)
    by_kind = {}
    for s in scenes:
        by_kind.setdefault(s["kind"], []).append(s)
    records = {"instruction": [], "start_pos": [], "target_pos": []}
    t_start = time.perf_counter()

    for instr in NOVEL_INSTRUCTIONS:
        for sc in by_kind["instruction"]:
            rec = run_episode(env, policy, pre, post, device, sc, args.max_steps,
                              SEED_BASE + int(sc["scene_id"]))
            records["instruction"].append(rec)
            print(f"[ACT-INS] instr#{NOVEL_INSTRUCTIONS.index(instr)} scene={rec['scene_id']} "
                  f"succ={rec['success']} dist {rec['dist_start']:.3f}->{rec['dist_end']:.3f}", flush=True)
    for sc in by_kind["start_pos"]:
        rec = run_episode(env, policy, pre, post, device, sc, args.max_steps,
                          SEED_BASE + 1000 + int(sc["scene_id"]))
        records["start_pos"].append(rec)
        print(f"[ACT-START] scene={rec['scene_id']} succ={rec['success']} "
              f"dist {rec['dist_start']:.3f}->{rec['dist_end']:.3f}", flush=True)
    for sc in by_kind["target_pos"]:
        rec = run_episode(env, policy, pre, post, device, sc, args.max_steps,
                          SEED_BASE + 2000 + int(sc["scene_id"]))
        records["target_pos"].append(rec)
        print(f"[ACT-TARGET] {sc.get('direction','?')} scene={rec['scene_id']} "
              f"succ={rec['success']} dist {rec['dist_start']:.3f}->{rec['dist_end']:.3f}", flush=True)
    env.close()

    summary = {k: summarize(v) for k, v in records.items()}
    summary["overall"] = summarize([r for v in records.values() for r in v])
    out = {
        "algorithm": "act", "checkpoint": args.checkpoint,
        "observation": "224x224 RGB + 13-d state (ACT; no language conditioning)",
        "protocol": {
            "scene_file": args.scenes, "scene_master_seed": payload.get("master_seed"),
            "max_steps": args.max_steps, "success_radius_m": SUCCESS_RADIUS,
            "seed_base": SEED_BASE,
            "instruction_note": "ACT has no language input; instruction test is fixed-scene repetition, not language generalization",
            "novel_instructions": NOVEL_INSTRUCTIONS,
        },
        "summary": summary, "per_test": records,
        "total_wall_s": round(time.perf_counter() - t_start, 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 68)
    for k in ("instruction", "start_pos", "target_pos", "overall"):
        s = summary[k]
        print(f"[ACT] {k}: {s['success_count']}/{s['n_episodes']} rate={s['success_rate']} "
              f"dist_imp={s['mean_dist_improvement']}")
    print("[ACT] raw ->", args.out, flush=True)
    print("ACT_GENERALIZATION_DONE")


if __name__ == "__main__":
    main()
