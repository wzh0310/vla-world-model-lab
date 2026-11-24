#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""样本效率实验统一评估器：PPO 对照组与世界模型想象策略同一评估流程。

在 mujoco 环境运行。两组各自使用训练时的最优观测口径：
  - PPO: obs_mode='rgb'(84x84灰度) + 4帧堆叠 + CnnPolicy（SB3 原生评估）
  - WM:  obs_mode='image'(demo 224) -> 64x64 -> ConvEncoder -> RSSM -> Actor
共同点：同一 PushCube 动力学、固定起点(randomize_start=False)、seed 0..N-1、
       每回合≤50步、同样记录 cumulative reward / final dist / success。

用法:
  python eval_unified_sfe.py --group ppo --step 200000 --episodes 30
  python eval_unified_sfe.py --group wm --pct 10 --episodes 30
  python eval_unified_sfe.py --all --episodes 30 --out <results.json>
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WMLAB = "/data/wzh/robot/vla-world-model-lab"
ZOO = "/data/wzh/robot/embodied-policy-zoo"
MUJOCO = "/data/wzh/robot/mujoco-studio-lab"
TARGET = np.array([0.82, -0.12], dtype=np.float32)
SUCCESS_TH = 0.03

os.environ.setdefault("MUJOCO_GL", "egl")


# ---------------- PPO 对照组 ----------------
def eval_ppo(step, n_episodes, seed0, max_steps):
    # PPO 评估走 embodied-policy-zoo 的统一环境（envs.paths + pushcube_envs）
    if ZOO not in sys.path:
        sys.path.insert(0, ZOO)
    from envs.paths import setup_sys_path, renderer_env
    for k, v in renderer_env().items():
        os.environ.setdefault(k, v)
    setup_sys_path()
    from stable_baselines3 import PPO
    import gymnasium as gym
    import pushcube_envs  # noqa
    path = f"/data/wzh/robot/outputs/ppo_vision/full_2m/checkpoints/vision_ppo_{step}_steps.zip"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = PPO.load(path, device="cuda")
    env = gym.make("PushCube-v0", obs_mode="rgb", rgb_preset="train",
                   randomize_start=False, disable_env_checker=True)
    rewards, dists, succs = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        frames = [obs] * 4
        total_r, success = 0.0, False
        for t in range(max_steps):
            st = np.concatenate(frames, axis=0)[None]
            a, _ = model.predict(st, deterministic=True)
            a = np.asarray(a).reshape(-1)
            obs, r, term, trunc, info = env.step(a)
            frames.append(obs); frames = frames[-4:]
            total_r += float(r)
            success = success or bool(info.get("success", False))
            if term or trunc:
                break
        cube = env.unwrapped.data.xpos[env.unwrapped._cube_body]
        dist = float(np.linalg.norm(cube[:2] - TARGET))
        rewards.append(total_r); dists.append(dist)
        succs.append(bool(success or dist < SUCCESS_TH))
    env.close()
    return _summary(rewards, dists, succs, n_episodes)


# ---------------- 世界模型实验组 ----------------
def load_wm_components(pct, device):
    sys.path.insert(0, WMLAB)
    from models.vae import ConvEncoder
    from models.rssm import RSSM
    from models.actor_critic import ActorCritic, ActionScaler

    pol = torch.load(f"{WMLAB}/outputs/sfe_policy_{pct}/imagine_policy.pt",
                     map_location=device, weights_only=False)
    wm = torch.load(f"{WMLAB}/outputs/sfe_wm_{pct}/checkpoints/wm_best.pt",
                    map_location=device, weights_only=False)
    a = wm.get("args", {})
    latent = int(a.get("latent_dim", 64))
    stoch = int(a.get("stoch_dim", 48))
    det = int(a.get("det_dim", 200))

    enc = ConvEncoder(latent_dim=latent).to(device)
    rssm = RSSM(stoch_dim=stoch, det_dim=det, action_dim=4, embed_dim=latent).to(device)
    sd = wm["model"]
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    rssm.load_state_dict({k[len("rssm."):]: v for k, v in sd.items() if k.startswith("rssm.")})
    ac = ActorCritic(feat_dim=pol["feat_dim"], act_dim=pol["act_dim"],
                     hidden=tuple(pol.get("hidden", [256, 256]))).to(device)
    ac.actor.load_state_dict(pol["actor"])
    ac.critic.load_state_dict(pol["critic"])
    scaler = ActionScaler.from_arrays(pol["action_low"], pol["action_high"], device=device)
    for m in (enc, rssm, ac):
        m.eval()
    return enc, rssm, ac, scaler, device


def obs_to_tensor(img, device):
    pil = Image.fromarray(img).resize((64, 64), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device).float()


def eval_wm(pct, n_episodes, seed0, max_steps, device="cpu"):
    # WM 评估走 mujoco-studio-lab 的 envs.push_cube_env（与训练/旧评估脚本一致）
    if "envs" in sys.modules:
        del sys.modules["envs"]
    while MUJOCO in sys.path:
        sys.path.remove(MUJOCO)
    sys.path.insert(0, MUJOCO)
    from envs.push_cube_env import PushCubeEnv
    enc, rssm, ac, scaler, device = load_wm_components(pct, device)
    env = PushCubeEnv(image_obs=True, max_episode_steps=max_steps,
                      obs_mode="image", rgb_preset="demo", randomize_start=False)
    rewards, dists, succs = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        state = rssm.initial_state(1, device=device)
        prev = torch.zeros(1, 4, device=device)
        total_r, success = 0.0, False
        with torch.no_grad():
            for t in range(max_steps):
                img = obs["image"] if isinstance(obs, dict) else obs
                embed = enc(obs_to_tensor(img, device))
                if isinstance(embed, (tuple, list)):
                    embed = embed[0]
                state, _ = rssm.observe(state, prev, embed)
                feat = rssm.get_feature(state)
                a_norm = ac.actor.sample(feat, deterministic=True)
                real = scaler.to_env(a_norm).cpu().numpy()[0]
                real = np.clip(real, env.action_space.low, env.action_space.high)
                obs, r, term, trunc, info = env.step(real)
                prev = a_norm
                total_r += float(r)
                success = success or bool(info.get("success", False))
                if term or trunc:
                    break
        cube = env.data.xpos[env._cube_body]
        dist = float(np.linalg.norm(cube[:2] - TARGET))
        rewards.append(total_r); dists.append(dist)
        succs.append(bool(success or dist < SUCCESS_TH))
    env.close()
    return _summary(rewards, dists, succs, n_episodes)


def _summary(rewards, dists, succs, n):
    return {
        "episodes": n,
        "success_rate": round(float(np.mean(succs)), 4),
        "mean_reward": round(float(np.mean(rewards)), 4),
        "std_reward": round(float(np.std(rewards)), 4),
        "mean_final_dist": round(float(np.mean(dists)), 4),
        "min_final_dist": round(float(np.min(dists)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["ppo", "wm"], default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--pct", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    PPO_STEPS = [200000, 600000, 1000000, 2000000]
    WM_PCTS = [10, 30, 50, 100]
    if args.all:
        tasks = [("ppo", s, None) for s in PPO_STEPS] + [("wm", None, p) for p in WM_PCTS]
    elif args.group == "ppo":
        tasks = [("ppo", args.step, None)]
    elif args.group == "wm":
        tasks = [("wm", None, args.pct)]
    else:
        raise SystemExit("需指定 --group 或 --all")

    results = []
    for group, step, pct in tasks:
        tag = f"PPO-vision-{step//1000}k" if group == "ppo" else f"WorldModel-imagine-{pct}%"
        try:
            if group == "ppo":
                m = eval_ppo(step, args.episodes, args.seed, args.max_steps)
                level = step
            else:
                m = eval_wm(pct, args.episodes, args.seed, args.max_steps, args.device)
                level = pct
            m.update({"group": group, "label": tag, "data_level": level})
            results.append(m)
            print(f"[SFE] {tag:26s} succ={m['success_rate']:.0%} "
                  f"reward={m['mean_reward']:+.4f} dist={m['mean_final_dist']:.4f}", flush=True)
            # 每档单独写 json（供汇总脚本归集）
            if args.out:
                single = args.out
                with open(single, "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback
            print(f"[SFE] 失败 {tag}: {e}", flush=True)
            traceback.print_exc()

    if args.out and len(results) > 1:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[SFE] 结果写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
