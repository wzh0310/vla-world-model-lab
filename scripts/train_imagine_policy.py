#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""想象轨迹训练（Dreamer 核心）：全程在世界模型内部训练策略，不接触真实环境。

训练循环（每步）：
  1) 从离线轨迹采样 B 个起点，用真实首帧逐帧 observe 建立后验初始状态（no_grad）；
  2) 冻结世界模型，用当前 Actor 在 RSSM 内展开 H 步纯想象：
       feature -> actor.sample() 得到动作与 log_prob/entropy
       -> rssm.imagine_step() 推进隐状态 -> reward_head 预测奖励
     Critic 对每步隐特征给出价值 V；
  3) 用 λ-return 计算目标回报 R^λ，优势 A = R^λ - V；
  4) PPO 风格裁剪策略损失 + 价值函数 MSE 损失 + 策略熵正则（鼓励探索）；
  5) 仅更新 Actor 与 Critic 参数，世界模型全程 requires_grad=False。

奖励口径：reward_head 输出为归一化尺度 clip(r/10,-1,1)，与训练一致。

用法（vla 环境）：
  python scripts/train_imagine_policy.py \
      --ckpt outputs/abl2_stoch48/checkpoints/wm_best.pt \
      --data outputs/data/trajectories.h5 --epochs 200 --smoke
  --smoke 只跑 1 个 batch（单步验证），--epochs 全量训练。
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
    ap = argparse.ArgumentParser(description="世界模型内想象轨迹训练（Dreamer 核心）")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/abl2_stoch48/checkpoints/wm_best.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/imagine_policy_full"))
    ap.add_argument("--horizon", type=int, default=10, help="想象步数 H（缩短以抑制复合误差）")
    ap.add_argument("--warmup", type=int, default=1, help="后验热身帧数 K")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200, help="训练 epoch 数")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.9, help="λ-return 的 λ（降低自举方差）")
    ap.add_argument("--lr-actor", type=float, default=1.0e-4)
    ap.add_argument("--lr-critic", type=float, default=1.0e-4, help="Critic 学习率（降低以稳定自举）")
    ap.add_argument("--clip-eps", type=float, default=0.2, help="PPO 裁剪系数")
    ap.add_argument("--entropy-coef", type=float, default=1.0e-2, help="熵正则系数（增大鼓励探索）")
    ap.add_argument("--value-coef", type=float, default=0.3, help="价值损失权重")
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--eval-interval", type=int, default=10, help="固定探针评估与最优保存间隔")
    ap.add_argument("--hubber-delta", type=float, default=1.0, help="Huber 损失阈值")
    ap.add_argument("--smoke", action="store_true", help="只跑 1 个 batch 的单步验证")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# ---------------------------------------------------------------- 数据预载
def preload_trajectories(hdf5_path, device):
    """预载全部轨迹到显存：归一化图像 + 动作，避免训练循环反复读盘。"""
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
    """采样 B 个起点：每条轨迹取前 K 帧 observe 得到后验初始状态。"""
    states = []
    for _ in range(B):
        k = keys[np.random.randint(len(keys))]
        imgs, acts = data[k]
        n = imgs.shape[0]
        st = int(np.random.randint(0, max(1, n - K)))
        state = wm.rssm.initial_state(1, device=wm.device)
        for t in range(st, st + K):
            embed = wm._encode(imgs[t].unsqueeze(0))             # (1, latent)
            state, _ = wm.rssm.observe(state, acts[t].unsqueeze(0), embed)
        states.append(state)
    det = torch.cat([s.det for s in states], 0)
    stoch = torch.cat([s.stoch for s in states], 0)
    return RSSMState(det=det, stoch=stoch)


# ---------------------------------------------------------------- 想象展开
def imagine_rollout(wm, ac, init_state, horizon, sample=True):
    """用当前 Actor 展开 H 步纯想象，返回逐步张量（保留梯度用于策略更新）。

    返回：
      features (B,H,F)   每步（动作前）隐特征
      log_probs(B,H)     采样动作的对数概率
      entropies(B,H)     动作分布熵
      rewards  (B,H)     奖励头预测
      actions  (B,H,A)   采样动作（detach，仅记录）
    """
    rssm = wm.rssm
    state = init_state
    feats, logps, ents, rews, acts = [], [], [], [], []
    for t in range(horizon):
        feat = rssm.get_feature(state)                          # (B,F)
        mu, log_std = ac.actor(feat)
        action = ac.actor.sample(feat, deterministic=not sample)  # (B,A)
        logp = ac.actor.log_prob(feat, action)                  # (B,)
        # 高斯分布熵（对角）：0.5 * sum(1 + log(2π σ²)) = 0.5*sum(1 + 2log_std + log(2π))，随后 /A 归一
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, device=feat.device))
        entropy = (0.5 * (1.0 + 2.0 * log_std + log_two_pi)).sum(-1) / ac.act_dim
        state, _ = rssm.imagine_step(state, action, sample=sample)
        reward = wm.reward_head(rssm.get_feature(state)).squeeze(-1) if wm.reward_head else torch.zeros_like(logp)
        feats.append(feat)
        logps.append(logp)
        ents.append(entropy)
        rews.append(reward)
        acts.append(action.detach())
    stack = lambda xs: torch.stack(xs, dim=1)
    # final_feat = V(s_H) 的输入，用作 λ-return 的 bootstrap
    final_feat = rssm.get_feature(state)                         # (B,F)
    return stack(feats), stack(logps), stack(ents), stack(rews), stack(acts), final_feat


def lambda_return(rewards, values, bootstrap, gamma, lam):
    """λ-return（Dreamer 标准）：从后向前递推，末步用 bootstrap = V(s_H)。

      R_t = r_t + γ[(1-λ)·V(s_{t+1}) + λ·R_{t+1}]
      其中 R_H := bootstrap，且 t=H-1 时 V(s_H)=bootstrap，
      故 R_{H-1} = r_{H-1} + γ·bootstrap。

    rewards/values: (B,H)，values[:,t] = V(s_t)。bootstrap: (B,)。
    返回 R (B,H)，无梯度（作为价值回归目标与优势基准）。
    """
    _, H = rewards.shape
    R = torch.zeros_like(rewards)
    next_value = bootstrap                                      # V(s_{t+1})，t=H-1 时为 V(s_H)
    last_return = bootstrap                                     # R_H
    for t in reversed(range(H)):
        last_return = rewards[:, t] + gamma * ((1.0 - lam) * next_value + lam * last_return)
        R[:, t] = last_return
        next_value = values[:, t]                               # 下一轮（t-1）的 V(s_t)
    return R.detach()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    wm = WorldModel.from_checkpoint(args.ckpt, device=str(dev))
    if wm.reward_head is None:
        raise SystemExit("检查点缺少 reward_head，无法进行想象训练")
    # 冻结世界模型全部参数
    for m in (wm.encoder, wm.decoder, wm.rssm, wm.reward_head, wm.value_head):
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

    ac = build_from_world_model(wm).to(dev)
    opt_actor = torch.optim.Adam(ac.actor.parameters(), lr=args.lr_actor)
    opt_critic = torch.optim.Adam(ac.critic.parameters(), lr=args.lr_critic)
    data, keys = preload_trajectories(args.data, dev)
    print(f"[IMAGTRAIN] 世界模型冻结；策略参数={ac.count_parameters():,} "
          f"轨迹={len(keys)} H={args.horizon} γ={args.gamma} λ={args.lam} "
          f"batch={args.batch_size} epochs={1 if args.smoke else args.epochs}", flush=True)

    # 固定探针起点：全程复用，用于无方差地评估策略进步 & 选最优权重
    probe_init = sample_init_states(wm, data, keys, args.batch_size, args.warmup)

    @torch.no_grad()
    def probe_return():
        """在固定起点上评估当前策略的平均想象回报（归一化尺度）。"""
        _, _, _, rw, _, _ = imagine_rollout(wm, ac, probe_init, args.horizon, sample=False)
        disc = torch.pow(args.gamma, torch.arange(args.horizon, device=rw.device, dtype=torch.float32))
        return float((rw * disc).sum(dim=1).mean())

    history = []
    t0 = time.time()
    best_probe, best_epoch = -1e9, 0
    total_epochs = 1 if args.smoke else args.epochs
    for ep in range(1, total_epochs + 1):
        init = sample_init_states(wm, data, keys, args.batch_size, args.warmup)
        features, logps, ents, rewards, actions, final_feat = imagine_rollout(
            wm, ac, init, args.horizon, sample=True)

        # 奖励裁剪：抑制奖励头在分布外状态产生的异常大值（发散主因之一）
        rewards = rewards.clamp(-1.0, 1.0)

        # Critic 价值（对每步隐特征）+ bootstrap V(s_H)
        values = ac.critic(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2])  # (B,H)
        bootstrap = ac.critic(final_feat)                        # (B,)

        # λ-return 目标 + 优势
        R = lambda_return(rewards, values.detach(), bootstrap.detach(), args.gamma, args.lam)   # (B,H)
        raw_adv = R - values.detach()
        # 优势标准化（减均值但不强制除到单位方差，避免放大噪声梯度）
        adv = (raw_adv - raw_adv.mean(dim=1, keepdim=True)) / (raw_adv.std(dim=1, keepdim=True) + 1.0)

        # 策略损失：Dreamer 标准 -logp·A，叠加 PPO 裁剪项约束更新幅度
        old_logps = logps.detach()
        ratio = torch.exp(logps - old_logps)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
        ppo_loss = -torch.min(surr1, surr2).mean()
        pg_loss = -(logps * adv).mean()                          # 主策略梯度信号
        policy_loss = 0.5 * pg_loss + 0.5 * ppo_loss
        entropy_loss = -args.entropy_coef * ents.mean()
        # Huber 价值损失：对 bootstrap 传播过来的异常回报更鲁棒
        value_loss = F.huber_loss(values, R, delta=args.hubber_delta)
        loss = policy_loss + args.value_coef * value_loss + entropy_loss

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        loss.backward()
        actor_gnorm = torch.nn.utils.clip_grad_norm_(ac.actor.parameters(), args.grad_clip)
        critic_gnorm = torch.nn.utils.clip_grad_norm_(ac.critic.parameters(), args.grad_clip)
        opt_actor.step()
        opt_critic.step()

        # 指标（想象奖励为归一化尺度）
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

        # 固定探针评估 + 保存最优策略权重
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
            print(f"[IMAGTRAIN] ep {ep:>4}/{total_epochs} loss={rec['loss']:.5f} "
                  f"policy={rec['policy_loss']:.5f} value={rec['value_loss']:.5f} "
                  f"entropy={rec['entropy']:.4f} rew={rec['imagined_reward']:.5f} "
                  f"a_gnorm={rec['actor_gnorm']:.4f} c_gnorm={rec['critic_gnorm']:.4f}{pv_tag}", flush=True)

    # 用最优权重作为最终策略（而非最后一轮，避免末端抖动）
    if best_epoch > 0 and not args.smoke:
        ac.actor.load_state_dict(best_actor_sd)
        ac.critic.load_state_dict(best_critic_sd)
        print(f"[IMAGTRAIN] 载入最优权重: epoch={best_epoch} probe_return={best_probe:.5f}", flush=True)

    # 保存策略权重（含结构参数，供 mujoco 环境评估加载）
    ckpt_out = {
        "actor": ac.actor.state_dict(),
        "critic": ac.critic.state_dict(),
        "best_epoch": best_epoch,
        "best_probe_return": round(best_probe, 5),
        "feat_dim": wm.feat_dim, "act_dim": ac.act_dim,
        "hidden": [256, 256],
        "action_low": [-0.02, -0.02, -0.02, 0.0],
        "action_high": [0.02, 0.02, 0.02, 1.0],
        "wm_ckpt": args.ckpt,
        "args": vars(args),
    }
    policy_path = os.path.join(args.out_dir, "imagine_policy.pt")
    torch.save(ckpt_out, policy_path)

    # 训练摘要
    n = len(history)
    k = max(1, n // 10)
    early_loss = float(np.mean([h["loss"] for h in history[:k]]))
    late_loss = float(np.mean([h["loss"] for h in history[-k:]]))
    early_rew = float(np.mean([h["imagined_reward"] for h in history[:k]]))
    late_rew = float(np.mean([h["imagined_reward"] for h in history[-k:]]))
    summary = {
        "epochs": n, "elapsed_sec": round(time.time() - t0, 1),
        "loss_early": round(early_loss, 6), "loss_late": round(late_loss, 6),
        "loss_down": bool(late_loss < early_loss),
        "reward_early": round(early_rew, 5), "reward_late": round(late_rew, 5),
        "reward_up": bool(late_rew > early_rew),
        "final": history[-1], "policy_path": policy_path,
    }
    with open(os.path.join(args.out_dir, "train_summary.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "history": history}, fp, ensure_ascii=False, indent=2)
    print(f"[IMAGTRAIN] 完成 {n} epoch 用时 {summary['elapsed_sec']}s | "
          f"loss {summary['loss_early']:.5f}->{summary['loss_late']:.5f} "
          f"(下降={summary['loss_down']}) | 想象奖励 {summary['reward_early']:.4f}->"
          f"{summary['reward_late']:.4f} (上升={summary['reward_up']})", flush=True)
    print(f"[IMAGTRAIN] 策略权重 -> {policy_path}", flush=True)
    print("IMAGINE_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
