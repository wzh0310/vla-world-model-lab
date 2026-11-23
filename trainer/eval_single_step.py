#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单步预测全面量化验证（第 9 周 Day7）。

在同一固定评估集上，对比三种"重建 / 预测"口径，并验证奖励预测精度：

1. 纯自编码器重建（AE）
   ae_best.pt 的 ConvVAE，图像 x_t -> 隐变量 mu -> 解码重建 x_t。
   这是不含任何动力学、只靠视觉压缩的参考基线（单帧重建上界）。

2. 世界模型后验重建（WM / posterior，上界）
   encoder(x_t) -> RSSM.observe(a_t, embed_t) -> 后验特征 -> decoder。
   "看到真实下一帧后"的重建，是单步预测理论上界。

3. 世界模型先验单步预测（WM / prior，真正预测）
   以真实前一帧建立的后验状态为起点，imagine_step(a_t) 纯动力学 -> decoder。
   不使用真实 x_t，是真正的"给定初始帧 + 动作 -> 预测下一帧"。

指标：PSNR(dB)、SSIM、MSE。奖励：MAE / RMSE（原始尺度 + 归一化尺度）。

评估集：覆盖全部轨迹，每条轨迹均匀取若干 20 步窗口，确定性且可复现；
归一化口径与训练（fast_loader）完全一致。注意当前为全量训练、未做轨迹级
held-out 划分，指标反映重建 / 预测保真度，不等同于严格泛化能力。

用法（vla 环境）：
  python trainer/eval_single_step.py --ckpt outputs/wm/checkpoints/wm_step003000.pt
"""
import argparse
import glob
import json
import math
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.vae import ConvVAE, ConvEncoder, ConvDecoder
from models.rssm import RSSM


def parse_args():
    ap = argparse.ArgumentParser(description="单步预测全面量化验证")
    ap.add_argument("--ckpt", default=None, help="世界模型检查点；默认选 step 最大的")
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/wm"))
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--windows-per-traj", type=int, default=4,
                    help="每条轨迹均匀取多少个窗口构成评估集")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


# ---------------------------------------------------------------- 数据加载
def load_eval_windows(hdf5_path, seq_len, image_size, windows_per_traj, device):
    """从 HDF5 构造确定性评估集，归一化口径与 fast_loader 一致。

    返回：
      imgs  (N,T,3,S,S) float16 [-1,1]
      acts  (N,T,4)     float32  [-1,1]
      rews  (N,T,1)     float32  [-1,1]（r/10 裁剪）
      rews_raw (N,T)    float32  原始奖励
    """
    t_load = __import__("time").time()
    with h5py.File(hdf5_path, "r") as f:
        low = torch.as_tensor(np.asarray(f.attrs["action_low"], np.float32), device=device)
        high = torch.as_tensor(np.asarray(f.attrs["action_high"], np.float32), device=device)
        span = torch.where(high == low, torch.ones_like(high), high - low)
        keys = sorted(f.keys())
        imgs, acts, rews, raws = [], [], [], []
        for k in keys:
            images = f[k]["images"][:]            # (n,224,224,3) uint8
            actions = f[k]["actions"][:]          # (n,4)
            rewards = f[k]["rewards"][:]          # (n,)
            n = images.shape[0]
            n_win = max(1, n - seq_len + 1)
            # 均匀取起点，覆盖整条轨迹且确定性
            if windows_per_traj >= n_win:
                starts = list(range(n_win))
            else:
                starts = np.linspace(0, n_win - 1, windows_per_traj).round().astype(int).tolist()
            raw_t = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
            small = F.interpolate(raw_t, size=(image_size, image_size),
                                  mode="bilinear", align_corners=False)
            small = (small / 127.5 - 1.0).half()  # (n,3,S,S)
            a_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
            a_t = ((a_t - low) / span * 2 - 1).clamp(-1, 1)
            r_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            r_norm = (r_t / 10.0).clamp(-1, 1)[:, None]
            for s in starts:
                imgs.append(small[s:s + seq_len].float())
                acts.append(a_t[s:s + seq_len])
                rews.append(r_norm[s:s + seq_len])
                raws.append(r_t[s:s + seq_len])
    print(f"[EVAL] 评估集: {len(imgs)} 序列 x {seq_len} 步, "
          f"覆盖 {len(keys)} 条轨迹, 加载 {__import__('time').time()-t_load:.1f}s", flush=True)
    return (torch.stack(imgs), torch.stack(acts),
            torch.stack(rews), torch.stack(raws))


# ---------------------------------------------------------------- 模型构建
def build_models(args, ckpt):
    """从检查点重建纯 AE 与世界模型各模块（均切到评估设备并设为 eval）。"""
    a = ckpt.get("args", {})
    latent = int(a.get("latent_dim", 64))
    stoch = int(a.get("stoch_dim", 64))
    det = int(a.get("det_dim", 200))
    feat_dim = det + stoch

    # 纯 AE：latent->latent 解码器
    ae = ConvVAE(latent_dim=latent).to(args.device)
    ae_sd = torch.load(args.ae_ckpt, map_location=args.device, weights_only=False)["model"]
    ae.load_state_dict(ae_sd)
    ae.eval()

    # 世界模型：encoder + decoder(264->img) + rssm + reward_head
    enc = ConvEncoder(latent_dim=latent).to(args.device)
    dec = ConvDecoder(latent_dim=feat_dim).to(args.device)
    rssm = RSSM(stoch_dim=stoch, det_dim=det, action_dim=4,
                embed_dim=latent).to(args.device)
    head = nn.Sequential(
        nn.Linear(feat_dim, 256), nn.SiLU(),
        nn.Linear(256, 128), nn.SiLU(),
        nn.Linear(128, 1),
    ).to(args.device)
    sd = ckpt["model"]
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    dec.load_state_dict({k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")})
    rssm.load_state_dict({k[len("rssm."):]: v for k, v in sd.items() if k.startswith("rssm.")})
    head.load_state_dict({k[len("reward_head."):]: v for k, v in sd.items() if k.startswith("reward_head.")})
    for m in (enc, dec, rssm, head):
        m.eval()
    print(f"[EVAL] 模型就绪: latent={latent} stoch={stoch} det={det} feat={feat_dim}", flush=True)
    return dict(ae=ae, enc=enc, dec=dec, rssm=rssm, head=head,
                latent=latent, stoch=stoch, det=det, feat_dim=feat_dim)


# ---------------------------------------------------------------- 指标函数
def psnr_from_mse(mse):
    """图像归一化到 [-1,1]，峰值平方 = 2^2 = 4。"""
    return 10.0 * math.log10(4.0 / (float(mse) + 1e-8))


def img_metrics(pred, target):
    """逐张计算 PSNR / SSIM，返回两个 numpy 数组（长度 = batch）。"""
    p = pred.detach().float().clamp(-1, 1).cpu().numpy()
    t = target.detach().float().clamp(-1, 1).cpu().numpy()
    B = p.shape[0]
    psnrs, ssims = np.zeros(B), np.zeros(B)
    for i in range(B):
        mse = np.mean((p[i] - t[i]) ** 2)
        psnrs[i] = 10.0 * math.log10(4.0 / (mse + 1e-8))
        ssims[i] = ssim(t[i], p[i], channel_axis=0, data_range=2.0)
    return psnrs, ssims


def summarize(arr):
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90))}


# ---------------------------------------------------------------- 主流程
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        cands = sorted(glob.glob(os.path.join(ROOT, "outputs/wm/checkpoints/wm_step*.pt")))
        if not cands:
            raise FileNotFoundError("没有世界模型检查点")
        ckpt_path = cands[-1]
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    print(f"[EVAL] 世界模型检查点: {ckpt_path} step={ckpt.get('step')}", flush=True)
    M = build_models(args, ckpt)
    imgs, acts, rews, raws = load_eval_windows(
        args.data, args.seq_len, args.image_size, args.windows_per_traj, args.device)

    N, T = imgs.shape[:2]
    S = args.image_size
    rec_ae_psnr, rec_ae_ssim = [], []
    rec_post_psnr, rec_post_ssim = [], []
    rec_prior_psnr, rec_prior_ssim = [], []
    rew_post_pred, rew_post_true = [], []
    rew_prior_pred, rew_prior_true = [], []

    batch = 16
    with torch.no_grad():
        for i0 in range(0, N, batch):
            x = imgs[i0:i0 + batch]            # (b,T,3,S,S)
            a = acts[i0:i0 + batch]            # (b,T,4)
            rr = raws[i0:i0 + batch]           # (b,T) 原始奖励
            b = x.shape[0]

            # (1) 纯 AE 重建：逐帧
            flat = x.reshape(b * T, 3, S, S)
            ae_rec = M["ae"](flat)["recon"].view(b, T, 3, S, S)

            # (2)(3) 世界模型：teacher-forcing 逐步（每步用真实 x_t 更新后验）
            emb = M["enc"](flat)[0].view(b, T, -1)
            state = M["rssm"].initial_state(b, device=args.device)
            post_recs, prior_recs = [], []
            post_rpred, prior_rpred = [], []
            # 收集每一步预测对应的真实奖励（原始尺度）
            post_true_t, prior_true_t = [], []
            for t in range(T):
                # 先验单步预测（用 t-1 的后验状态 + a_t），t=0 无前驱
                if t >= 1:
                    ps, _ = M["rssm"].imagine_step(state, a[:, t])
                    pf = M["rssm"].get_feature(ps)
                    prior_recs.append(M["dec"](pf))
                    prior_rpred.append(M["head"](pf).squeeze(-1))   # (b,)
                    prior_true_t.append(rr[:, t])                    # (b,)
                # 后验重建（观察真实 x_t）
                state, _ = M["rssm"].observe(state, a[:, t], emb[:, t])
                feat = M["rssm"].get_feature(state)
                post_recs.append(M["dec"](feat))
                post_rpred.append(M["head"](feat).squeeze(-1))       # (b,)
                post_true_t.append(rr[:, t])                          # (b,)

            # t=1..T-1 三种口径对齐比较（同一批真实帧 x[:, 1:]）
            sl = slice(1, T)
            target = x[:, sl].reshape(-1, 3, S, S)
            p_ae, s_ae = img_metrics(ae_rec[:, sl].reshape(-1, 3, S, S), target)
            rec_ae_psnr.extend(p_ae.tolist()); rec_ae_ssim.extend(s_ae.tolist())

            prior_stack = torch.stack(prior_recs, dim=1)             # (b,T-1,3,S,S)
            p_pri, s_pri = img_metrics(prior_stack.reshape(-1, 3, S, S), target)
            rec_prior_psnr.extend(p_pri.tolist()); rec_prior_ssim.extend(s_pri.tolist())

            post_stack = torch.stack(post_recs, dim=1)[:, sl]        # (b,T-1,3,S,S)
            p_pos, s_pos = img_metrics(post_stack.reshape(-1, 3, S, S), target)
            rec_post_psnr.extend(p_pos.tolist()); rec_post_ssim.extend(s_pos.tolist())

            # 奖励预测误差（原始尺度）：pred 与 true 均为 (b,T-1) 展平
            post_pred_mat = torch.stack(post_rpred, dim=1)[:, sl]     # (b,T-1)
            post_true_mat = torch.stack(post_true_t, dim=1)[:, sl]    # (b,T-1)
            prior_pred_mat = torch.stack(prior_rpred, dim=1)          # (b,T-1)
            prior_true_mat = torch.stack(prior_true_t, dim=1)         # (b,T-1)
            rew_post_pred.extend(post_pred_mat.cpu().numpy().ravel().tolist())
            rew_post_true.extend(post_true_mat.cpu().numpy().ravel().tolist())
            rew_prior_pred.extend(prior_pred_mat.cpu().numpy().ravel().tolist())
            rew_prior_true.extend(prior_true_mat.cpu().numpy().ravel().tolist())

    def rew_stats(pred, true):
        pred = np.asarray(pred, np.float32); true = np.asarray(true, np.float32)
        e = pred - true
        return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                "bias": float(np.mean(e)), "n": int(len(e))}

    result = {
        "checkpoint": os.path.relpath(ckpt_path, ROOT),
        "step": ckpt.get("step"),
        "eval_set": {"sequences": N, "steps_each": T, "frames_scored": T - 1,
                     "traj_covered": "全部轨迹，每条均匀取窗口",
                     "note": "全量训练未做轨迹级 held-out；指标为重建/预测保真度"},
        "image_metrics": {
            "ae_recon": {"psnr": summarize(rec_ae_psnr), "ssim": summarize(rec_ae_ssim)},
            "wm_posterior": {"psnr": summarize(rec_post_psnr), "ssim": summarize(rec_post_ssim)},
            "wm_prior_single_step": {"psnr": summarize(rec_prior_psnr), "ssim": summarize(rec_prior_ssim)},
        },
        "reward_metrics": {
            "posterior": rew_stats(rew_post_pred, rew_post_true),
            "prior": rew_stats(rew_prior_pred, rew_prior_true),
        },
    }

    # 保存 JSON
    out_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "single_step_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[EVAL] 指标已保存: {json_path}", flush=True)

    # 打印摘要
    im = result["image_metrics"]
    print("\n===== 图像重建 / 预测（t=1..T-1 对齐）=====", flush=True)
    for name, label in [("ae_recon", "纯AE重建"), ("wm_posterior", "WM后验重建(上界)"),
                        ("wm_prior_single_step", "WM先验单步预测")]:
        print(f"  {label:20s} PSNR={im[name]['psnr']['mean']:.2f}±{im[name]['psnr']['std']:.2f} dB "
              f"SSIM={im[name]['ssim']['mean']:.4f}±{im[name]['ssim']['std']:.4f}", flush=True)
    rm = result["reward_metrics"]
    print("\n===== 奖励预测（原始尺度）=====", flush=True)
    for name, label in [("posterior", "后验奖励"), ("prior", "先验奖励")]:
        print(f"  {label:12s} MAE={rm[name]['mae']:.4f} RMSE={rm[name]['rmse']:.4f} "
              f"bias={rm[name]['bias']:.4f} (n={rm[name]['n']})", flush=True)


if __name__ == "__main__":
    main()
