#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""消融与泛化的统一评估器：在 mujoco 环境加载想象策略跑真实环境回合。

两种模式：
  --mode standard    : 标准评估，randomize_start=True，N 回合（用于步长/奖励消融）
  --mode generalize  : 泛化评估，读取 generalization_scenes.json 的 50 个未见场景
                       （立方体起点/目标位置 + 视觉 jitter 变化），逐场景评估

关键：成功判定一律用 env.target_xy（场景目标，由 _apply_scene 设置），
      不能硬编码 [0.82,-0.12]，否则泛化场景目标变化会导致误判。

与项目已有泛化基准（eval_generalization.py）口径一致：每场景最多 50 步，
场景顺序逐字节一致，成功=立方体到「当前场景目标」距离<0.03m。

用法（mujoco 环境）：
  python scripts/eval_ablation.py --policy outputs/abl_h5/imagine_policy.pt \
      --mode standard --episodes 30 --out outputs/abl_h5/eval.json
  python scripts/eval_ablation.py --policy outputs/imagine_policy_v2/imagine_policy.pt \
      --mode generalize --out outputs/gen_imagine/eval.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")

from models.vae import ConvEncoder
from models.rssm import RSSM
from models.actor_critic import ActorCritic, ActionScaler

SCENES_PATH = "/data/wzh/robot/embodied-policy-zoo/configs/generalization_scenes.json"
SEED_BASE = 900000
SUCCESS_RADIUS = 0.03


def parse_args():
    ap = argparse.ArgumentParser(description="消融/泛化统一评估（想象策略零样本迁移）")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--mode", choices=["standard", "generalize"], default="standard")
    ap.add_argument("--episodes", type=int, default=30, help="standard 模式的回合数")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="结果 JSON 输出路径")
    ap.add_argument("--deterministic", action="store_true")
    return ap.parse_args()


def load_components(policy_path, device):
    p = torch.load(policy_path, map_location=device, weights_only=False)
    wm_ckpt = p["wm_ckpt"]
    if not os.path.isabs(wm_ckpt):
        wm_ckpt = os.path.join(ROOT, wm_ckpt)
    wm = torch.load(wm_ckpt, map_location=device, weights_only=False)
    a = wm.get("args", {})
    latent = int(a.get("latent_dim", 64))
    stoch = int(a.get("stoch_dim", 48))
    det = int(a.get("det_dim", 200))

    enc = ConvEncoder(latent_dim=latent).to(device)
    rssm = RSSM(stoch_dim=stoch, det_dim=det, action_dim=4, embed_dim=latent).to(device)
    sd = wm["model"]
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    rssm.load_state_dict({k[len("rssm."):]: v for k, v in sd.items() if k.startswith("rssm.")})

    ac = ActorCritic(feat_dim=p["feat_dim"], act_dim=p["act_dim"],
                     hidden=tuple(p.get("hidden", [256, 256]))).to(device)
    ac.actor.load_state_dict(p["actor"])
    ac.critic.load_state_dict(p["critic"])
    scaler = ActionScaler.from_arrays(p["action_low"], p["action_high"], device=device)
    for m in (enc, rssm, ac):
        m.eval()
    return enc, rssm, ac, scaler, device


def obs_to_tensor(img, device):
    pil = Image.fromarray(img).resize((64, 64), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return t.float()


@torch.no_grad()
def run_episode(env, enc, rssm, ac, scaler, device, seed, max_steps, deterministic, scene=None):
    if scene is not None:
        obs, info = env.reset(options={"scene": scene})
    else:
        obs, info = env.reset(seed=seed)
    state = rssm.initial_state(1, device=device)
    total_reward, steps = 0.0, 0
    prev_act_norm = torch.zeros(1, 4, device=device)
    for t in range(max_steps):
        img = obs["image"] if isinstance(obs, dict) else obs
        embed = enc(obs_to_tensor(img, device))
        if isinstance(embed, (tuple, list)):
            embed = embed[0]
        state, _ = rssm.observe(state, prev_act_norm, embed)
        feat = rssm.get_feature(state)
        act_norm = ac.actor.sample(feat, deterministic=deterministic)
        real = scaler.to_env(act_norm).cpu().numpy()[0]
        real = np.clip(real, env.action_space.low, env.action_space.high)
        obs, reward, term, trunc, info = env.step(real)
        total_reward += float(reward)
        steps += 1
        prev_act_norm = act_norm
        if term or trunc:
            break
    # 成功判定：用当前场景的目标 env.target_xy（泛化场景目标会变化）
    cube = env.data.xpos[env._cube_body]
    target = np.asarray(env.target_xy, dtype=np.float64)
    dist = float(np.linalg.norm(cube[:2] - target[:2]))
    success = bool(dist < SUCCESS_RADIUS)
    return {"success": success, "steps": int(steps), "dist": round(dist, 4),
            "total_reward": round(total_reward, 3),
            "scene_id": int(scene["scene_id"]) if scene else None}


def summarize(results):
    n = len(results)
    succ = [r for r in results if r["success"]]
    dists = [r["dist"] for r in results]
    return {
        "n": n,
        "success_rate": round(len(succ) / n, 4) if n else 0.0,
        "mean_reward": round(float(np.mean([r["total_reward"] for r in results])), 4) if n else None,
        "mean_final_dist": round(float(np.mean(dists)), 4) if n else None,
        "min_final_dist": round(float(np.min(dists)), 4) if n else None,
        "mean_steps": round(float(np.mean([r["steps"] for r in results])), 2) if n else None,
    }


def main():
    args = parse_args()
    # 关键：CPU 推理时限制 torch 线程数为 1，避免多线程上下文切换反而拖慢逐步推理
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    from envs.push_cube_env import PushCubeEnv
    enc, rssm, ac, scaler, device = load_components(args.policy, args.device)
    print(f"[EVALABL] 模型就绪 feat={ac.feat_dim} 设备={device} 模式={args.mode}", flush=True)

    results = []
    if args.mode == "standard":
        env = PushCubeEnv(obs_mode="image", rgb_preset="demo", randomize_start=True,
                          max_episode_steps=args.max_steps)
        for ep in range(args.episodes):
            rec = run_episode(env, enc, rssm, ac, scaler, device,
                              seed=args.seed + ep, max_steps=args.max_steps,
                              deterministic=args.deterministic)
            results.append(rec)
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"[EVALABL] ep {ep+1}/{args.episodes} succ={rec['success']} "
                      f"reward={rec['total_reward']:.3f}", flush=True)
        env.close()
    else:
        scenes = json.load(open(SCENES_PATH, encoding="utf-8"))["scenes"]
        env = PushCubeEnv(obs_mode="image", rgb_preset="demo", scene_override=True,
                          max_episode_steps=args.max_steps)
        for sc in scenes:
            torch.manual_seed(SEED_BASE + int(sc["scene_id"]))
            np.random.seed(SEED_BASE + int(sc["scene_id"]))
            rec = run_episode(env, enc, rssm, ac, scaler, device,
                              seed=SEED_BASE + int(sc["scene_id"]), max_steps=args.max_steps,
                              deterministic=args.deterministic, scene=sc)
            results.append(rec)
            sid = int(sc["scene_id"])
            if (sid + 1) % 10 == 0 or sid == 0:
                print(f"[EVALABL] scene {sid+1}/{len(scenes)} succ={rec['success']} "
                      f"dist={rec['dist']:.4f} reward={rec['total_reward']:.3f}", flush=True)
        env.close()

    summary = summarize(results)
    payload = {"mode": args.mode, "policy": args.policy, "summary": summary,
               "results": results}
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(f"[EVALABL] 汇总 succ={summary['success_rate']:.0%} "
          f"reward={summary['mean_reward']:.3f} dist={summary['mean_final_dist']:.4f} "
          f"-> {args.out}", flush=True)
    print("EVAL_ABLATION_OK", flush=True)


if __name__ == "__main__":
    main()
