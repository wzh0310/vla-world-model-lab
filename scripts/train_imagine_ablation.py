#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""想象轨迹训练的消融版本：在原训练逻辑上新增「奖励来源」开关，供奖励预测器消融。

与 train_imagine_policy.py 的唯一差异：
  --reward-mode wm   : 用世界模型 reward_head 的预测作为想象奖励（默认，正常训练）
  --reward-mode zero : 想象奖励恒为 0（消融「无奖励预测器」——策略失去任务目标信号）

注意：探针评估（probe_return）在两种模式下都使用真实 reward_head 作为统一「裁判」，
      因此 probe 曲线的差异能公平反映「用奖励训练 vs 不用奖励训练」所得策略的质量。

其余训练逻辑（λ-return / PPO+Huber / 冻结世界模型 / 最优权重）与原脚本完全一致。

用法（vla 环境）：
  python scripts/train_imagine_ablation.py \
      --ckpt outputs/abl2_stoch48/checkpoints/wm_best.pt \
      --data outputs/data/trajectories.h5 --horizon 12 \
      --reward-mode zero --out-dir outputs/abl_no_reward
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.world_model import WorldModel
from models.rssm import RSSMState
from models.actor_critic import build_from_world_model


def parse_args():
    ap = argparse.ArgumentParser(description="世界模型内想象训练（消融版：奖励来源可控）")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/abl2_stoch48/checkpoints/wm_best.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon", type=int, default=12, help="想象步数 H（步长消融变量）")
    ap.add_argument("--reward-mode", choices=["wm", "zero"], default="wm",
                    help=" imagination 奖励来源：wm=奖励头预测；zero=零奖励（无奖励预测器消融）")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--lr-actor", type=float, default=1.5e-4)
    ap.add_argument("--lr-critic", type=float, default=1.0e-4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=2.0e-2)
    ap.add_argument("--value-coef", type=float, default=0.3)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--hubber-delta", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def preload_trajectories(hdf5_path, device):
    data = {}
    with h5py.File(hdf5_path, "r") as f:
        low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
        high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
        span = torch.where(high == low, torch.ones_like(high), high - low)
        for k in f.keys():
            if not isinstance(f[k], h5py.Group):
                continue
            raw = torch.from_numpy(f[k]["images"][:]).permute(0, 3, 1, 2).float().to(device)
            small = F.interpolate(raw, size=(64, 64), mode="bilinear", align_corners=False)
            imgs = small / 127.5 - 1.0
            a = torch.as_tensor(f[k]["actions"][:], dtype=torch.float32, device=device)
            acts = ((a - low) / span * 2.0 - 1.0).clamp(-1.0, 1.0)
            data[k] = (imgs, acts)
    return data, list(data.keys())


@torch.no_grad()
def sample_init_states(wm, data, keys, B, K):
    states = []
    for _ in range(B):
        k = keys[np.random.randint(len(keys))]
        imgs, acts = data[k]
        n = imgs.shape[0]
        st = int(np.random.randint(0, max(1, n - K)))
        state = wm.rssm.initial_state(1, device=wm.device)
        for t in range(st, st + K):
            embed = wm._encode(imgs[t].unsqueeze(0))
            state, _ = wm.rssm.observe(state, acts[t].unsqueeze(0), embed)
        states.append(state)
    det = torch.cat([s.det for s in states], 0)
    stoch = torch.cat([s.stoch for s in states], 0)
    return RSSMState(det=det, stoch=stoch)


def imagine_rollout(wm, ac, init_state, horizon, sample=True, use_reward=True):
    """展开 H 步想象。use_reward=False 时奖励恒零（消融用）。"""
    rssm = wm.rssm
    state = init_state
    feats, logps, ents, rews, acts = [], [], [], [], []
    for t in range(horizon):
        feat = rssm.get_feature(state)
        mu, log_std = ac.actor(feat)
        action = ac.actor.sample(feat, deterministic=not sample)
        logp = ac.actor.log_prob(feat, action)
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, device=feat.device))
        entropy = (0.5 * (1.0 + 2.0 * log_std + log_two_pi)).sum(-1) / ac.act_dim
        state, _ = rssm.imagine_step(state, action, sample=sample)
        if use_reward and wm.reward_head is not None:
            reward = wm.reward_head(rssm.get_feature(state)).squeeze(-1)
        else:
            reward = torch.zeros_like(logp)
        feats.append(feat)
        logps.append(logp)
        ents.append(entropy)
        rews.append(reward)
        acts.append(action.detach())
    stack = lambda xs: torch.stack(xs, dim=1)
    final_feat = rssm.get_feature(state)
    return stack(feats), stack(logps), stack(ents), stack(rews), stack(acts), final_feat


def lambda_return(rewards, values, bootstrap, gamma, lam):
    _, H = rewards.shape
    R = torch.zeros_like(rewards)
    next_value = bootstrap
    last_return = bootstrap
    for t in reversed(range(H)):
        last_return = rewards[:, t] + gamma * ((1.0 - lam) * next_value + lam * last_return)
        R[:, t] = last_return
        next_value = values[:, t]
    return R.detach()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    wm = WorldModel.from_checkpoint(args.ckpt, device=str(dev))
    if wm.reward_head is None:
        raise SystemExit("检查点缺少 reward_head，无法进行统一探针评估")
    for m in (wm.encoder, wm.decoder, wm.rssm, wm.reward_head, wm.value_head):
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

    ac = build_from_world_model(wm).to(dev)
    opt_actor = torch.optim.Adam(ac.actor.parameters(), lr=args.lr_actor)
    opt_critic = torch.optim.Adam(ac.critic.parameters(), lr=args.lr_critic)
    data, keys = preload_trajectories(args.data, dev)
    use_reward = args.reward_mode == "wm"
    print(f"[ABLTRAIN] 世界模型冻结 | 策略参数={ac.count_parameters():,} 轨迹={len(keys)} "
          f"H={args.horizon} reward_mode={args.reward_mode}(训练用奖励={use_reward}) "
          f"γ={args.gamma} λ={args.lam} epochs={1 if args.smoke else args.epochs}", flush=True)

    probe_init = sample_init_states(wm, data, keys, args.batch_size, args.warmup)

    @torch.no_grad()
    def probe_return():
        # 探针始终用真实奖励头（统一裁判），不受 reward_mode 影响
        _, _, _, rw, _, _ = imagine_rollout(wm, ac, probe_init, args.horizon,
                                            sample=False, use_reward=True)
        disc = torch.pow(args.gamma, torch.arange(args.horizon, device=rw.device, dtype=torch.float32))
        return float((rw * disc).sum(dim=1).mean())

    history = []
    t0 = time.time()
    best_probe, best_epoch = -1e9, 0
    total_epochs = 1 if args.smoke else args.epochs
    for ep in range(1, total_epochs + 1):
        init = sample_init_states(wm, data, keys, args.batch_size, args.warmup)
        features, logps, ents, rewards, actions, final_feat = imagine_rollout(
            wm, ac, init, args.horizon, sample=True, use_reward=use_reward)
        rewards = rewards.clamp(-1.0, 1.0)

        values = ac.critic(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2])
        bootstrap = ac.critic(final_feat)
        R = lambda_return(rewards, values.detach(), bootstrap.detach(), args.gamma, args.lam)
        raw_adv = R - values.detach()
        adv = (raw_adv - raw_adv.mean(dim=1, keepdim=True)) / (raw_adv.std(dim=1, keepdim=True) + 1.0)

        old_logps = logps.detach()
        ratio = torch.exp(logps - old_logps)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
        ppo_loss = -torch.min(surr1, surr2).mean()
        pg_loss = -(logps * adv).mean()
        policy_loss = 0.5 * pg_loss + 0.5 * ppo_loss
        entropy_loss = -args.entropy_coef * ents.mean()
        value_loss = F.huber_loss(values, R, delta=args.hubber_delta)
        loss = policy_loss + args.value_coef * value_loss + entropy_loss

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        loss.backward()
        actor_gnorm = torch.nn.utils.clip_grad_norm_(ac.actor.parameters(), args.grad_clip)
        critic_gnorm = torch.nn.utils.clip_grad_norm_(ac.critic.parameters(), args.grad_clip)
        opt_actor.step()
        opt_critic.step()

        rec = {
            "epoch": ep,
            "loss": round(float(loss.detach()), 6),
            "policy_loss": round(float(policy_loss.detach()), 6),
            "value_loss": round(float(value_loss.detach()), 6),
            "entropy": round(float(ents.detach().mean()), 4),
            "imagined_reward": round(float(rewards.detach().sum(dim=1).mean()), 5),
            "actor_gnorm": round(float(actor_gnorm), 5),
            "critic_gnorm": round(float(critic_gnorm), 5),
        }
        history.append(rec)
        if ep % args.eval_interval == 0 or ep == 1:
            pv = probe_return()
            rec["probe_return"] = round(pv, 5)
            if pv > best_probe:
                best_probe, best_epoch = pv, ep
                import copy
                best_actor_sd = copy.deepcopy(ac.actor.state_dict())
                best_critic_sd = copy.deepcopy(ac.critic.state_dict())
        if ep == 1 or ep % args.log_interval == 0 or args.smoke:
            pv_tag = f" probe={rec.get('probe_return', 0):.4f}" if ep == 1 or ep % args.eval_interval == 0 else ""
            print(f"[ABLTRAIN] ep {ep:>4}/{total_epochs} loss={rec['loss']:.5f} "
                  f"value={rec['value_loss']:.5f} entropy={rec['entropy']:.4f} "
                  f"rew={rec['imagined_reward']:.5f} a_gnorm={rec['actor_gnorm']:.4f}{pv_tag}", flush=True)

    if best_epoch > 0 and not args.smoke:
        ac.actor.load_state_dict(best_actor_sd)
        ac.critic.load_state_dict(best_critic_sd)
        print(f"[ABLTRAIN] 载入最优权重: epoch={best_epoch} probe={best_probe:.5f}", flush=True)

    ckpt_out = {
        "actor": ac.actor.state_dict(), "critic": ac.critic.state_dict(),
        "best_epoch": best_epoch, "best_probe_return": round(best_probe, 5),
        "feat_dim": wm.feat_dim, "act_dim": ac.act_dim,
        "hidden": [256, 256],
        "action_low": [-0.02, -0.02, -0.02, 0.0],
        "action_high": [0.02, 0.02, 0.02, 1.0],
        "wm_ckpt": args.ckpt, "args": vars(args),
    }
    policy_path = os.path.join(args.out_dir, "imagine_policy.pt")
    torch.save(ckpt_out, policy_path)

    n = len(history)
    k = max(1, n // 10)
    early_rew = float(np.mean([h["imagined_reward"] for h in history[:k]]))
    late_rew = float(np.mean([h["imagined_reward"] for h in history[-k:]]))
    early_probe = float(np.mean([h["probe_return"] for h in history[:k] if "probe_return" in h])) if any("probe_return" in h for h in history[:k]) else None
    late_probe = float(np.mean([h["probe_return"] for h in history[-k:] if "probe_return" in h])) if any("probe_return" in h for h in history[-k:]) else None
    summary = {
        "epochs": n, "elapsed_sec": round(time.time() - t0, 1),
        "reward_mode": args.reward_mode, "horizon": args.horizon,
        "reward_early": round(early_rew, 5), "reward_late": round(late_rew, 5),
        "probe_early": round(early_probe, 5) if early_probe is not None else None,
        "probe_late": round(late_probe, 5) if late_probe is not None else None,
        "best_probe_return": round(best_probe, 5), "best_epoch": best_epoch,
        "policy_path": policy_path,
    }
    with open(os.path.join(args.out_dir, "train_summary.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "history": history}, fp, ensure_ascii=False, indent=2)
    print(f"[ABLTRAIN] 完成 {n} epoch 用时 {summary['elapsed_sec']}s | "
          f"reward_mode={args.reward_mode} probe {summary['probe_early']}->{summary['probe_late']} "
          f"(best={best_probe:.5f}@{best_epoch})", flush=True)
    print("ABLATION_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
