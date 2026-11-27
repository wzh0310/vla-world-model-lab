# -*- coding: utf-8 -*-
"""跨任务（左/右）ACT 评估（lerobot 环境），沿用已验证的 ACT 加载与推理逻辑。"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in ("/data/wzh/robot/embodied-policy-zoo",
          "/data/wzh/robot/embodied-policy-zoo/scripts",
          "/data/wzh/robot/mujoco-studio-lab",
          "/data/wzh/robot/mujoco-studio-lab/envs"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pushcube_envs.push_cube_env import PushCubeEnv  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.envs import make_env_config  # noqa: E402
import lerobot_push_cube_env  # noqa: E402,F401

SUCCESS_RADIUS = 0.03
INSTR = {"left": "把方块推到左边", "right": "把方块推到右边"}


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
    dist0 = float(info["cube_target_dist"])
    dmin = dist0
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
        d = float(info["cube_target_dist"])
        dmin = min(dmin, d)
        if term or trunc:
            break
    dist_end = float(info["cube_target_dist"])
    return {
        "task": scene["task"], "scene_id": int(scene["scene_id"]),
        "instruction": INSTR[scene["task"]],
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
        out[task] = {"n_episodes": n, "success_count": succ,
                     "success_rate": round(succ / n, 4) if n else 0.0,
                     "mean_dist_improvement": round(float(np.mean(
                         [r["dist_start"] - r["dist_end"] for r in rs])) if n else 0.0, 4)}
    n = len(records); succ = sum(r["success"] for r in records)
    out["overall"] = {"n_episodes": n, "success_count": succ,
                      "success_rate": round(succ / n, 4) if n else 0.0,
                      "mean_dist_improvement": round(float(np.mean(
                          [r["dist_start"] - r["dist_end"] for r in records])) if n else 0.0, 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=60)
    args = ap.parse_args()
    payload = json.load(open(args.scenes, encoding="utf-8"))
    scenes = payload["scenes"]
    policy, pre, post, device = load_policy(args.checkpoint)
    env = PushCubeEnv(scene_override=True, obs_mode="both", rgb_preset="demo",
                      max_episode_steps=args.max_steps)
    records = []
    for sc in scenes:
        rec = run_episode(env, policy, pre, post, device, sc, args.max_steps,
                          900000 + int(sc["scene_id"]))
        records.append(rec)
        print(f"[ACT-{sc['task']}] scene={rec['scene_id']} succ={rec['success']} "
              f"dist {rec['dist_start']:.3f}->{rec['dist_end']:.3f} min={rec['dist_min']:.3f}", flush=True)
    env.close()
    summary = summarize(records)
    out = {"algorithm": "act", "checkpoint": args.checkpoint,
           "scenes_file": args.scenes, "max_steps": args.max_steps,
           "success_radius_m": SUCCESS_RADIUS, "instructions": INSTR,
           "note": "ACT 视觉+13维状态，无语言条件；指令仅任务语义标记",
           "summary": summary, "episodes": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for k in ("left", "right", "overall"):
        s = summary[k]
        print(f"[ACT] {k}: {s['success_count']}/{s['n_episodes']} "
              f"rate={s['success_rate']} imp={s['mean_dist_improvement']}", flush=True)
    print("CROSS_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
