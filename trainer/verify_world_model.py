#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""世界模型单步预测验证：初始帧 + 动作 -> 想象下一步 -> 解码预测下一帧。

两种预测对比：
  后验重建：observe(真实 x_{t+1}) 得后验 -> 解码（模型"看到"真值后的重建，上界）
  先验预测：imagine_step(a_t) 纯动力学 -> 解码（真正的单步预测，无真实下一帧）
并计算两者的 MSE / PSNR，保存可视化对比图。

用法：python trainer/verify_world_model.py --ckpt outputs/wm/checkpoints/wm_step003000.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, ROOT)
from fast_loader import GPUSequenceSampler
from models.vae import ConvEncoder, ConvDecoder
from models.rssm import RSSM


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="检查点；默认选 step 最大的")
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/data/trajectories.h5"))
    ap.add_argument("--ae-ckpt", default=os.path.join(ROOT, "outputs/checkpoints/ae_best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/wm/vis/prediction.png"))
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def build_model(args, ckpt):
    a = ckpt.get('args', {})
    latent = a.get('latent_dim', 64)
    stoch = a.get('stoch_dim', 64)
    det = a.get('det_dim', 200)
    enc = ConvEncoder(latent_dim=latent)
    dec = ConvDecoder(latent_dim=det + stoch)
    rssm = RSSM(stoch_dim=stoch, det_dim=det, action_dim=4, embed_dim=latent)
    head = None  # 奖励头不参与图像预测
    ae = torch.load(args.ae_ckpt, map_location=args.device, weights_only=False)
    enc.load_state_dict({k[len('encoder.'):]: v for k, v in ae['model'].items() if k.startswith('encoder.')})
    dec.deconv.load_state_dict({k[len('decoder.deconv.'):]: v for k, v in ae['model'].items()
                                if k.startswith('decoder.deconv.')})
    sdim = ckpt['model']
    enc = enc.to(args.device); dec = dec.to(args.device); rssm = rssm.to(args.device)
    enc.load_state_dict({k[len('encoder.'):]: v for k, v in sdim.items() if k.startswith('encoder.')})
    dec.load_state_dict({k[len('decoder.'):]: v for k, v in sdim.items() if k.startswith('decoder.')})
    rssm.load_state_dict({k[len('rssm.'):]: v for k, v in sdim.items() if k.startswith('rssm.')})
    enc.eval(); dec.eval(); rssm.eval()
    return (enc, dec, rssm), latent


def psnr(mse):
    return 10 * np.log10(4.0 / (mse + 1e-8))


def main():
    args = parse_args()
    ckpt_path = args.ckpt
    if ckpt_path is None:
        cands = sorted(glob.glob(os.path.join(ROOT, 'outputs/wm/checkpoints/wm_step*.pt')))
        if not cands:
            raise FileNotFoundError('没有世界模型检查点')
        ckpt_path = cands[-1]
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    print(f'[VERIFY] 检查点: {ckpt_path} step={ckpt.get("step")}')
    (enc, dec, rssm), latent = build_model(args, ckpt)

    sampler = GPUSequenceSampler(args.data, seq_len=20, image_size=64,
                                 device=args.device, seed=123)
    imgs, acts, _ = sampler.sample(args.num_samples, with_rewards=True)
    B, T = imgs.shape[:2]
    with torch.no_grad():
        # t=0：用真实 x_0 建立后验状态（起点）
        e0, _ = enc(imgs[:, 0])
        state, _ = rssm.observe(rssm.initial_state(B, device=args.device), acts[:, 0], e0)
        # t=1：先纯动力学想象，再用真实 x_1 后验对比
        prior_state, _ = rssm.imagine_step(state, acts[:, 1])
        pred_prior = dec(rssm.get_feature(prior_state))           # 先验预测 x_1
        e1, _ = enc(imgs[:, 1])
        post_state, _ = rssm.observe(state, acts[:, 1], e1)
        recon_post = dec(rssm.get_feature(post_state))            # 后验重建 x_1

        target = imgs[:, 1]
        mse_prior = F.mse_loss(pred_prior, target).item()
        mse_post = F.mse_loss(recon_post, target).item()
        print(f'[VERIFY] 单步先验预测: MSE={mse_prior:.5f} PSNR={psnr(mse_prior):.2f}dB')
        print(f'[VERIFY] 后验重建(上界): MSE={mse_post:.5f} PSNR={psnr(mse_post):.2f}dB')

        # 可视化：上排真实 x_0 / 中排真实 x_1 / 下排先验预测 / 末排后验重建
        rows = [imgs[:, 0], target, pred_prior, recon_post]
        grid = make_grid(torch.cat(rows, dim=0), nrow=B, normalize=True,
                         value_range=(-1, 1), pad_value=0.3)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        save_image(grid, args.out)
        print(f'[VERIFY] 对比图: {args.out}  (上=初始帧 中=真实下一帧 下=先验预测 末=后验重建)')


if __name__ == '__main__':
    main()
