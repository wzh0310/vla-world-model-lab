#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""想象策略训练最小 Demo：验证「世界模型 → 想象轨迹 → 策略更新」链路。

不追求策略性能，只验证链路可跑通、可微梯度能回传到策略、预测奖励总和是否有上升趋势。

流程（每轮迭代）：
  1) 从真实轨迹采样若干起点，用真实首帧 + observe 建立后验初始状态（此步用 no_grad，
     不作为想象的一部分；想象起点用真实帧编码是 Dreamer 标准做法）；
  2) 之后 H 步全程在模型内想象：policy(feature) -> action -> rssm.imagine_step（不接真实图像）
     -> reward_head(feature) 预测每步奖励；
  3) 折扣回报 R = sum gamma^t * r_t；loss = -R.mean()；
  4) 世界模型参数冻结（requires_grad=False）但仍参与计算图，梯度经世界模型回传只更新策略。

关键：imagine_step 的 GRU/prior 对 action 可微，reward_head 对 feature 可微，
因此 -R 的梯度能沿"奖励→隐状态→动作→策略参数"完整回传——这是本 Demo 要验证的核心。

用法（vla 环境）：
  python trainer/imagine_policy.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trainer"))
from eval_single_step import build_models
from models.policy import MLPPolicy


def parse_args():
    ap = argparse.ArgumentParser(description="世界模型内想象策略训练最小 Demo")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/wm_tune1/checkpoints/wm_step004000.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/imagine_policy"))
    ap.add_argument("--trajs", default="traj_0002,traj_0007,traj_0011,traj_0015,traj_0020,traj_0001,traj_0005")
    ap.add_argument("--horizon", type=int, default=15, help="想象步数（Day11 结论：10~16 步）")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--sample", action="store_true",
                    help="想象时从先验重参数化采样；默认关闭用先验均值（确定性，曲线更平滑）")
    ap.add_argument("--fixed-init", action="store_true",
                    help="固定同一批初始状态训练（消除起点方差，便于观察策略优化趋势）")
    ap.add_argument("--baseline", action="store_true",
                    help="用价值头预测作 baseline，训练优势 return-V（降低方差，Dreamer 做法）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def preload_trajectories(hdf5_path, keys, device):
    """把指定轨迹的图像/动作/奖励预载到显存，训练循环直接切片。"""
    data = {}
    with h5py.File(hdf5_path, "r") as f:
        low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
        high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
        span = torch.where(high == low, torch.ones_like(high), high - low)
        for k in keys:
            if k not in f:
                continue
            raw = torch.from_numpy(f[k]["images"][:]).permute(0, 3, 1, 2).float().to(device)
            small = F.interpolate(raw, size=(64, 64), mode="bilinear", align_corners=False)
            imgs = (small / 127.5 - 1.0).half()
            a = torch.as_tensor(f[k]["actions"][:], dtype=torch.float32, device=device)
            acts = ((a - low) / span * 2 - 1).clamp(-1, 1)
            r = torch.as_tensor(f[k]["rewards"][:], dtype=torch.float32, device=device)
            rews = (r / 10.0).clamp(-1, 1)
            data[k] = (imgs, acts, rews)
    return data


def imagine_return(M, policy, init_state, horizon, gamma, sample=True, actions=None,
                   return_feats=False):
    """在模型内展开 H 步想象，返回折扣回报 (B,) 与每步奖励 (B,H)。

    actions 若为 None 则每步用 policy(feature) 生成动作；否则用给定动作序列
    （探针用），用于对比"专家动作"的想象回报上界。
    return_feats=True 时额外返回每步初始特征 (B,H,F)，供 baseline 计算。
    """
    rssm, head = M["rssm"], M["head"]
    state = init_state
    rewards, feats = [], []
    for t in range(horizon):
        feat = rssm.get_feature(state)
        feats.append(feat)
        if actions is not None:
            act = actions[:, t]
        else:
            act = policy(feat)
        state, _ = rssm.imagine_step(state, act, sample=sample)
        r = head(rssm.get_feature(state)).squeeze(-1)
        rewards.append(r)
    rewards = torch.stack(rewards, dim=1)                       # (B,H)
    disc = torch.pow(gamma, torch.arange(horizon, device=rewards.device, dtype=torch.float32))
    ret = (rewards * disc).sum(dim=1)                           # (B,)
    if return_feats:
        return ret, rewards, torch.stack(feats, dim=1)          # (B,H,F)
    return ret, rewards


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    M = build_models(args, ckpt)
    enc, rssm, head = M["enc"], M["rssm"], M["head"]

    # 可选：加载价值头作 baseline（build_models 只加载 reward head）
    value_head = None
    if args.baseline:
        from models.heads import ValueHead
        feat_dim = M["feat_dim"]
        vh = ValueHead(feat_dim).to(args.device)
        sd = ckpt["model"]
        vh.load_state_dict({k[len("value_head."):]: v for k, v in sd.items()
                            if k.startswith("value_head.")})
        vh.eval()
        for p in vh.parameters():
            p.requires_grad_(False)
        value_head = vh
        print("[IMAG] 价值头已加载作 baseline", flush=True)

    # 冻结世界模型：保留计算图（梯度可穿过），但不更新其参数
    frozen = [M["ae"], enc, M["dec"], rssm, head]
    if value_head is not None:
        frozen.append(value_head)
    for m in frozen:
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    policy = MLPPolicy(feat_dim=M["feat_dim"], act_dim=4).to(args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"[IMAG] 世界模型已冻结；策略参数量={policy.count_parameters():,} "
          f"horizon={args.horizon} gamma={args.gamma}", flush=True)

    keys = [t.strip() for t in args.trajs.split(",") if t.strip()]
    data = preload_trajectories(args.data, keys, args.device)
    keys = [k for k in keys if k in data]
    print(f"[IMAG] 预载 {len(keys)} 条轨迹", flush=True)

    def sample_init_states(B):
        """采样 B 个起点，用真实首帧 + observe 建立后验初始状态（no_grad）。"""
        states, expert_acts = [], []
        with torch.no_grad():
            for _ in range(B):
                k = keys[np.random.randint(len(keys))]
                imgs, acts, _ = data[k]
                n = imgs.shape[0]
                st = int(np.random.randint(2, max(3, n - args.horizon - 2)))
                e0 = enc(imgs[st].float().unsqueeze(0))[0]
                init = rssm.observe(rssm.initial_state(1, device=args.device),
                                    acts[st].unsqueeze(0), e0)[0]
                states.append(init)
                expert_acts.append(acts[st + 1:st + 1 + args.horizon].unsqueeze(0))
        det = torch.cat([s.det for s in states], dim=0)
        st = torch.cat([s.stoch for s in states], dim=0)
        from models.rssm import RSSMState
        return RSSMState(det=det, stoch=st), torch.cat(expert_acts, dim=0)

    # 固定起点模式：预先采样一批初始状态，全程复用（消除起点方差，凸显策略学习）
    eval_init = None
    if args.fixed_init:
        eval_init, _ = sample_init_states(args.batch_size)
        print("[IMAG] 固定初始状态模式：全程复用同一批起点", flush=True)

    # ---------------- 探针：奖励信号是否可用于训练 ----------------
    with torch.no_grad():
        init_p, exp_a = sample_init_states(args.batch_size)
        rand_ret, _ = imagine_return(M, policy, init_p, args.horizon, args.gamma, sample=args.sample)
        expert_ret, _ = imagine_return(M, policy, init_p, args.horizon, args.gamma,
                                       sample=args.sample, actions=exp_a)
        probe = {
            "random_policy_return_mean": round(float(rand_ret.mean()), 5),
            "expert_actions_return_mean": round(float(expert_ret.mean()), 5),
        }
    print(f"[PROBE] 随机策略想象回报={probe['random_policy_return_mean']:.5f} "
          f"专家动作想象回报={probe['expert_actions_return_mean']:.5f} (归一化奖励尺度)",
          flush=True)

    # 评估函数：在给定初始状态上算当前策略的想象回报（用于趋势曲线，no_grad）
    def evaluate(init):
        with torch.no_grad():
            r, _ = imagine_return(M, policy, init, args.horizon, args.gamma, sample=args.sample)
        return float(r.mean())

    # ---------------- 想象策略训练 ----------------
    hist = []
    for it in range(1, args.iters + 1):
        init = eval_init if args.fixed_init else sample_init_states(args.batch_size)[0]
        if value_head is not None:
            ret, rew, feats = imagine_return(M, policy, init, args.horizon, args.gamma,
                                             sample=args.sample, return_feats=True)
            # baseline：价值头对每步特征的价值预测，构造优势（不含梯度，仅作方差缩减）
            with torch.no_grad():
                baselines = value_head(feats.reshape(-1, feats.shape[-1])).squeeze(-1).reshape(feats.shape[:2])
            disc = torch.pow(args.gamma, torch.arange(args.horizon, device=rew.device, dtype=torch.float32))
            advantage = ((rew - baselines) * disc).sum(dim=1)
            loss = -advantage.mean()
        else:
            ret, rew = imagine_return(M, policy, init, args.horizon, args.gamma, sample=args.sample)
            loss = -ret.mean()
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        opt.step()
        if it == 1 or it % 10 == 0:
            # 用固定 init 评估（若有），否则用本轮 ret（带方差，如实记录）
            trend_val = evaluate(eval_init) if args.fixed_init else float(ret.mean().detach())
            hist.append({"iter": it, "return_mean": round(trend_val, 5),
                         "batch_return_mean": round(float(ret.mean().detach()), 5),
                         "reward_sum_mean": round(float(rew.sum(dim=1).mean().detach()), 5),
                         "grad_norm": round(float(gnorm), 4)})
            tag = "eval_fixed" if args.fixed_init else "batch"
            print(f"[IMAG] iter {it:>3}/{args.iters} {tag}_return={trend_val:.5f} "
                  f"batch_return={ret.mean().item():.5f} gnorm={gnorm.item():.4f}",
                  flush=True)

    # 训练后探针：固定起点模式用同一批 eval_init；否则固定种子重建同批起点
    if args.fixed_init:
        end_ret_mean = evaluate(eval_init)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        with torch.no_grad():
            init_end, _ = sample_init_states(args.batch_size)
            end_ret, _ = imagine_return(M, policy, init_end, args.horizon, args.gamma, sample=args.sample)
        end_ret_mean = float(end_ret.mean())
    probe["trained_policy_return_mean"] = round(end_ret_mean, 5)

    # 趋势判定：首 10% 迭代 vs 末 10% 迭代的 return 均值
    k = max(1, len(hist) // 10)
    early = np.mean([h["return_mean"] for h in hist[:k]])
    late = np.mean([h["return_mean"] for h in hist[-k:]])
    trend = {
        "early_mean": round(float(early), 5), "late_mean": round(float(late), 5),
        "delta": round(float(late - early), 5),
        "rising": bool(late > early),
    }

    result = {
        "ckpt": args.ckpt, "horizon": args.horizon, "gamma": args.gamma,
        "iters": args.iters, "batch_size": args.batch_size, "lr": args.lr,
        "probe": probe, "trend": trend, "history": hist,
    }
    met_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(met_dir, exist_ok=True)
    with open(os.path.join(met_dir, "imagine_policy.json"), "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    # 趋势折线图（英文标签）
    vis_dir = os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    its = [h["iter"] for h in hist]
    rets = [h["return_mean"] for h in hist]
    plt.figure(figsize=(8, 4.5))
    plt.plot(its, rets, color="#d62728", lw=1.8, marker="o", ms=3, label="Imagined return")
    plt.axhline(probe["random_policy_return_mean"], color="gray", ls=":", lw=1.2,
                label="Random policy (probe)")
    plt.axhline(probe["expert_actions_return_mean"], color="#2ca02c", ls="--", lw=1.2,
                label="Expert actions (probe)")
    plt.xlabel("Training iteration")
    plt.ylabel("Imagined discounted return (reward scale normalized)")
    plt.title("Imagined-policy training: return vs iteration")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "imagine_return_curve.png"), dpi=130)
    plt.close()
    print(f"[IMAG] 趋势: early={trend['early_mean']:.5f} late={trend['late_mean']:.5f} "
          f"delta={trend['delta']:+.5f} rising={trend['rising']}", flush=True)
    print(f"[IMAG] OK -> {met_dir}/imagine_policy.json", flush=True)


if __name__ == "__main__":
    main()
