# -*- coding: utf-8 -*-
"""世界模型统一评估指标库：像素级 / 结构级 / 感知级 / 动力学级。

本文件是指标实现的唯一来源；``trainer/metrics.py`` 保留为兼容转发层。

四类核心指标（对应评估体系设计）：
  像素级   MSE、PSNR          —— 逐像素精度
  结构级   SSIM               —— 视觉结构一致性
  感知级   PerceptualMetric   —— 深度特征感知距离（人眼相似度代理）
  动力学级 latent_drift_l2    —— 开环隐状态相对 teacher-forcing 后验的每步漂移
           reward_mae         —— 奖励预测平均绝对误差

感知级说明：LPIPS 原版（alexnet/vgg）预训练权重因外网受限不可下载，本库用容器内
已缓存的 torchvision ResNet18 ImageNet 权重实现同属"感知级"的特征余弦距离
（有界 [0,2]，越小越相似）。接口与 LPIPS 对齐（distance(pred, target)），未来
联网后可无缝替换为 lpips.LPIPS。
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ImageNet 归一化常数（ResNet18 感知特征用）
_IN_MEAN = [0.485, 0.456, 0.406]
_IN_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- 像素级
def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """逐样本 MSE，输入 (N,3,H,W) 与 [-1,1] 范围，返回 (N,)。"""
    p = pred.detach().float()
    t = target.detach().float()
    return torch.mean((p - t) ** 2, dim=tuple(range(1, p.ndim))).cpu().numpy()


def psnr_from_mse(mse: float, peak: float = 2.0) -> float:
    """[-1,1] 范围图像峰值 = 2。"""
    return 10.0 * float(np.log10(peak ** 2 / (mse + 1e-8)))


def psnr_metric(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return np.array([psnr_from_mse(float(m)) for m in mse_metric(pred, target)])


# ---------------------------------------------------------------- 结构级
def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """逐样本 SSIM（channel_axis=2 的 HWC 口径，data_range=2）。"""
    from skimage.metrics import structural_similarity as _ssim
    p = pred.detach().float().clamp(-1, 1).cpu().numpy().transpose(0, 2, 3, 1)
    t = target.detach().float().clamp(-1, 1).cpu().numpy().transpose(0, 2, 3, 1)
    return np.array([_ssim(t[i], p[i], channel_axis=2, data_range=2.0)
                     for i in range(p.shape[0])])


# ---------------------------------------------------------------- 感知级
class PerceptualMetric:
    """深度特征感知距离（ResNet18 ImageNet 特征余弦距离，有界 [0,2]）。

    替代说明见模块 docstring。distance(pred, target) -> (N,)，越小越相似。
    """

    def __init__(self, device: torch.device, batch_size: int = 16):
        import torchvision
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
        self.net = nn.Sequential(*list(net.children())[:-2]).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        mean = torch.tensor(_IN_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IN_STD, device=device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std
        for p in self.net.parameters():
            p.requires_grad_(False)

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float().clamp(-1, 1)
        x = (x / 2 + 0.5 - self._mean) / self._std
        return x

    @torch.no_grad()
    def distance(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """pred/target: (N,3,H,W) [-1,1] -> (N,) 感知距离。"""
        N = pred.shape[0]
        out = np.zeros(N, dtype=np.float64)
        for s in range(0, N, self.batch_size):
            e = min(s + self.batch_size, N)
            f1 = self.net(self._pre(pred[s:e].to(self.device)))
            f2 = self.net(self._pre(target[s:e].to(self.device)))
            f1 = f1.flatten(1)
            f2 = f2.flatten(1)
            cos = nn.functional.cosine_similarity(f1, f2, dim=1)
            out[s:e] = (1.0 - cos).cpu().numpy()
        return out


# ---------------------------------------------------------------- 动力学级
def latent_drift_l2(feats_open_loop: torch.Tensor,
                    feats_teacher_forcing: torch.Tensor) -> np.ndarray:
    """隐状态每步预测误差：开环特征相对 teacher-forcing 后验特征的归一化 L2。

    参数：两个 (N, F) 序列（同一时刻对齐，同一动作序列）。
    返回 (N,)：每步 ||f_ol - f_tf|| / sqrt(F)。值越大说明动力学漂移越重。
    """
    if feats_open_loop.shape != feats_teacher_forcing.shape:
        raise ValueError(f"特征序列形状不一致: {tuple(feats_open_loop.shape)} vs "
                         f"{tuple(feats_teacher_forcing.shape)}")
    F = feats_open_loop.shape[-1]
    d = torch.norm(feats_open_loop - feats_teacher_forcing, dim=-1) / np.sqrt(F)
    return d.cpu().numpy()


def reward_mae(rewards_pred: torch.Tensor, rewards_target: torch.Tensor) -> float:
    """奖励预测 MAE（标量，序列整体平均）。"""
    return float(torch.mean(torch.abs(rewards_pred - rewards_target)))


def summarize_by_step(per_step: dict) -> dict:
    """逐 step 指标 dict -> 汇总（mean/min/max）。"""
    return {k: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                "max": float(np.max(v))} for k, v in per_step.items()
            if v is not None and len(v) > 0}


if __name__ == "__main__":
    # 自检
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.rand(4, 3, 64, 64) * 2 - 1
    b = (a * 0.9).clamp(-1, 1)
    print("mse", mse_metric(a, b).round(4).tolist())
    print("psnr", psnr_metric(a, b).round(2).tolist())
    print("ssim", ssim_metric(a, b).round(3).tolist())
    pm = PerceptualMetric(dev)
    print("perceptual", pm.distance(a, b).round(4).tolist())
    print("perceptual(identical)", pm.distance(a, a).round(4).tolist())
    f1, f2 = torch.randn(10, 264), torch.randn(10, 264)
    print("latent_drift", latent_drift_l2(f1, f2).round(3).tolist()[:3])
    print("SELF_TEST_OK")
