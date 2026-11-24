#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零样本迁移评估：把想象训练得到的策略在 PushCube 真实环境跑回合。

在 mujoco 环境运行（有 mujoco/gymnasium/torch/PIL）。策略推理链路：
  观测图像(224) -> resize 64 -> ConvEncoder 编码 -> RSSM observe 跟踪后验隐状态
  -> Actor 采样动作(归一化[-1,1]) -> ActionScaler 映射回真实动作范围 -> env.step

世界模型权重来自想象训练所用的检查点（与策略配套），全程不微调、不接触训练数据，
纯粹检验「世界模型内学到的策略能否直接迁移到真实动力学」。

用法（mujoco 环境）：
  python scripts/eval_imagine_policy.py \
      --policy outputs/imagine_policy_full/imagine_policy.pt \
      --episodes 20
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/data/wzh/robot/mujoco-studio-lab")

from models.vae import ConvEncoder
from models.rssm import RSSM
from models.actor_critic import ActorCritic, ActionScaler


def parse_args():
    ap = argparse.ArgumentParser(description="想象策略零样本迁移评估（PushCube 真实环境）")
    ap.add_argument("--policy", default=os.path.join(ROOT, "outputs/imagine_policy_full/imagine_policy.pt"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="默认与策略同目录")
    ap.add_argument("--deterministic", action="store_true", help="用动作均值而非采样")
    return ap.parse_args()


def load_components(policy_path, device):
    """加载策略权重 + 重建配套的世界模型 encoder/rssm（与训练同一 ckpt）。"""
    p = torch.load(policy_path, map_location=device, weights_only=False)
    wm_ckpt = p["wm_ckpt"]
    if not os.path.isabs(wm_ckpt):
        wm_ckpt = os.path.join(ROOT, wm_ckpt)
    wm = torch.load(wm_ckpt, map_location=device, weights_only=False)
    a = wm.get("args", {})
    latent = int(a.get("latent_dim", 64))
    stoch = int(a.get("stoch_dim", 48))
    det = int(a.get("det_dim", 200))
    feat = det + stoch

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
    return enc, rssm, ac, scaler, latent, device


def obs_to_tensor(img, device):
    """(H,W,3) uint8 -> (1,3,64,64) 归一化 [-1,1]，与训练预处理一致。"""
    pil = Image.fromarray(img).resize((64, 64), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 127.5 - 1.0     # [-1,1]
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return t.float()


@torch.no_grad()
def run_episode(env, enc, rssm, ac, scaler, latent, device, seed, max_steps, deterministic):
    obs, info = env.reset(seed=seed)
    state = rssm.initial_state(1, device=device)
    total_reward, steps, success = 0.0, 0, False
    prev_act_norm = torch.zeros(1, 4, device=device)
    for t in range(max_steps):
        img = obs["image"] if isinstance(obs, dict) else obs
        embed = enc(obs_to_tensor(img, device))                 # (1,latent) 编码器返回 (mu, logvar)?
        if isinstance(embed, (tuple, list)):
            embed = embed[0]
        # 用上一动作 + 当前帧嵌入做后验推断，跟踪真实环境隐状态
        state, _ = rssm.observe(state, prev_act_norm, embed)
        feat = rssm.get_feature(state)                          # (1,F)
        act_norm = ac.actor.sample(feat, deterministic=deterministic)  # (1,4)
        real = scaler.to_env(act_norm).cpu().numpy()[0]         # 映射回真实动作范围
        real = np.clip(real, env.action_space.low, env.action_space.high)
        obs, reward, term, trunc, info = env.step(real)
        total_reward += float(reward)
        steps += 1
        prev_act_norm = act_norm
        if term or trunc:
            break
    # 成功判定（与 eval_act.py 口径一致）：立方体到目标中心距离 < 0.03m
    cube = env.data.xpos[env._cube_body]
    dist = float(np.linalg.norm(cube[:2] - np.array([0.82, -0.12])))
    success = bool(dist < 0.03)
    steps = int(info.get("steps", steps))
    return {"success": success, "steps": steps, "dist": round(dist, 4),
            "total_reward": round(total_reward, 3)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.policy))
    os.makedirs(out_dir, exist_ok=True)

    from envs.push_cube_env import PushCubeEnv
    enc, rssm, ac, scaler, latent, device = load_components(args.policy, args.device)
    print(f"[EVALPOL] 模型就绪: latent={latent} feat={ac.feat_dim} 设备={device} "
          f"回合={args.episodes} max_steps={args.max_steps}", flush=True)

    env = PushCubeEnv(image_obs=True, max_episode_steps=args.max_steps,
                      obs_mode="image", rgb_preset="demo", randomize_start=True)
    results = []
    for ep in range(args.episodes):
        rec = run_episode(env, enc, rssm, ac, scaler, latent, device,
                          seed=args.seed + ep, max_steps=args.max_steps,
                          deterministic=args.deterministic)
        results.append(rec)
        print(f"[EVALPOL] ep {ep+1:>2}/{args.episodes} success={rec['success']} "
              f"steps={rec['steps']} reward={rec['total_reward']:.3f}", flush=True)
    env.close()

    successes = [r for r in results if r["success"]]
    dists = [r.get("dist", float("nan")) for r in results]
    summary = {
        "episodes": args.episodes,
        "success_rate": round(len(successes) / args.episodes, 4),
        "avg_steps_all": round(float(np.mean([r["steps"] for r in results])), 2),
        "avg_steps_success": round(float(np.mean([r["steps"] for r in successes])), 2) if successes else None,
        "avg_total_reward": round(float(np.mean([r["total_reward"] for r in results])), 3),
        "avg_final_dist": round(float(np.nanmean(dists)), 4),
        "min_final_dist": round(float(np.nanmin(dists)), 4),
        "deterministic": args.deterministic,
    }
    with open(os.path.join(out_dir, "eval_real_env.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "episodes_detail": results}, fp, ensure_ascii=False, indent=2)
    print(f"[EVALPOL] 汇总: 成功率={summary['success_rate']:.0%} "
          f"成功步数均值={summary['avg_steps_success']} "
          f"平均累计奖励={summary['avg_total_reward']:.3f}", flush=True)
    print("EVAL_IMAGINE_POLICY_OK", flush=True)


if __name__ == "__main__":
    main()
