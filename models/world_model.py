# -*- coding: utf-8 -*-
"""世界模型统一推理封装：从检查点一键加载，提供 rollout() 多步预测管线。

rollout() 是「初始帧 + 动作序列 → 多步预测」的单一入口，三段流程：
  1) 后验热身（可选）：把前 K 帧真实图像逐帧 observe 进 RSSM，得到准确的起点状态；
  2) 开环想象：imagine_actions 纯动力学自回归展开 N 步，全程不接入任何真实帧；
  3) 同步输出：预测图像序列、RSSM 隐状态序列、拼接特征序列、奖励/价值预测序列，
     供下游策略训练（trainer/imagine_policy.py）与评估直接消费。

动作对齐约定（与数据集/训练口径一致）：动作按「到达帧」索引——a[t] 与 img[t]
同索引配对，observe(s_{t-1}, a_t, e_t) 把状态推进到帧 t。

用法（vla 环境）：
    from models.world_model import WorldModel
    wm = WorldModel.from_checkpoint("outputs/wm_tune1/checkpoints/wm_step004000.pt")
    out = wm.rollout_from_trajectory(images, actions, warmup_steps=3, horizon=10)
    out["images"]   # (N,3,64,64) 预测帧
    out["features"] # (N,F) 预测隐特征（det||stoch）
    out["rewards"]  # (N,) 预测奖励（归一化尺度）
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# 支持两种执行方式：包内导入（from models.world_model import WorldModel）
# 与直接执行自检（python models/world_model.py）——后者需先把仓库根目录加入 sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.vae import ConvEncoder, ConvDecoder
from models.factory import build_dynamics
from models.reward_predictor import RewardHead, ValueHead


class WorldModel:
    """推理态世界模型：encoder + RSSM + decoder + reward_head + value_head。"""

    def __init__(self, encoder: ConvEncoder, decoder: ConvDecoder, rssm: nn.Module,
                 reward_head: Optional[nn.Module], value_head: Optional[nn.Module],
                 latent_dim: int, stoch_dim: int, det_dim: int, image_size: int,
                 device: torch.device, dynamics: str = "rssm"):
        self.encoder = encoder
        self.decoder = decoder
        self.rssm = rssm
        self.reward_head = reward_head
        self.value_head = value_head
        self.latent_dim = latent_dim
        self.stoch_dim = stoch_dim
        self.det_dim = det_dim
        self.feat_dim = det_dim + stoch_dim
        self.image_size = image_size
        self.device = device
        self.dynamics = dynamics
        for m in (encoder, decoder, rssm, reward_head, value_head):
            if m is not None:
                m.eval()

    # ---------------------------------------------------------------- 加载
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: Optional[str] = None) -> "WorldModel":
        """从世界模型检查点重建全部模块（权重自包含，不依赖外部 AE 检查点）。

        旧检查点若无 value_head，则 value_head=None，rollout 不输出价值序列。
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        a = ckpt.get("args", {})
        latent = int(a.get("latent_dim", 64))
        stoch = int(a.get("stoch_dim", 64))
        det = int(a.get("det_dim", 200))
        image_size = int(a.get("image_size", 64) or 64)
        dynamics = str(a.get("dynamics", "rssm") or "rssm")
        feat = det + stoch

        enc = ConvEncoder(latent_dim=latent).to(device)
        dec = ConvDecoder(latent_dim=feat, out_size=image_size).to(device)
        rssm, feat = build_dynamics(dynamics, feat, 4, latent, stoch, det)
        rssm = rssm.to(device)
        rew = RewardHead(feat).to(device)
        val = ValueHead(feat).to(device)

        sd = ckpt["model"]

        def strip(prefix, module):
            keys = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not keys:
                return None
            module.load_state_dict(keys)
            return module

        enc = strip("encoder.", enc) or enc
        dec = strip("decoder.", dec) or dec
        rssm = strip("rssm.", rssm) or rssm
        rew = strip("reward_head.", rew)
        val = strip("value_head.", val)

        wm = cls(enc, dec, rssm, rew, val, latent, stoch, det, image_size, device,
                 dynamics=dynamics)
        wm.ckpt_args = a
        print(f"[WM] 已加载 {ckpt_path}: dynamics={dynamics} latent={latent} stoch={stoch} det={det} "
              f"feat={feat} image={image_size} "
              f"reward_head={'有' if rew is not None else '无'} "
              f"value_head={'有' if val is not None else '无'}", flush=True)
        return wm

    # ---------------------------------------------------------------- 内部工具
    def _check_image(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 3:                       # 单帧 (3,H,W) -> (1,3,H,W)
            img = img.unsqueeze(0)
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"图像应为 (K,3,H,W)，得到 {tuple(img.shape)}")
        s = self.image_size
        if img.shape[-2] != s or img.shape[-1] != s:
            raise ValueError(f"图像分辨率应为 {s}x{s}（与检查点一致），得到 {tuple(img.shape[-2:])}")
        return img.to(self.device)

    def _check_actions(self, act: torch.Tensor) -> torch.Tensor:
        if act.ndim == 1:
            act = act.unsqueeze(0)
        if act.ndim != 2 or act.shape[-1] != 4:
            raise ValueError(f"动作应为 (T,4)，得到 {tuple(act.shape)}")
        return act.to(self.device)

    def _encode(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim == 3:                      # 单帧 (3,H,W) -> (1,3,H,W)
            imgs = imgs.unsqueeze(0)
        mu, _ = self.encoder(imgs)
        return mu                               # 用后验均值作嵌入（推理确定性）

    # ---------------------------------------------------------------- 核心接口
    @torch.no_grad()
    def rollout(self, context_images: torch.Tensor, context_actions: torch.Tensor,
                imagine_actions: torch.Tensor, sample: bool = True) -> Dict:
        """多步预测管线。

        参数：
          context_images:  (K,3,S,S) 真实帧 t_0..t_{K-1}（K>=1）
          context_actions: (K,4)     与各帧同索引配对的动作（按到达帧索引）
          imagine_actions: (N,4)     从帧 t_{K-1} 状态出发的想象动作，到达帧 t_K..t_{K+N-1}
          sample:          想象步是否从先验采样；False 用先验均值（确定性，便于复现）

        返回 dict：
          images   (N,3,S,S)   预测帧序列
          states   List[RSSMState] 长度 N 的隐状态序列
          features (N,F)       拼接特征（det||stoch）
          rewards  (N,)        奖励头预测（归一化尺度；无奖励头时为 None）
          values   (N,)        价值头预测（无价值头时为 None）
        """
        ctx_img = self._check_image(context_images)
        ctx_act = self._check_actions(context_actions)
        img_act = self._check_actions(imagine_actions)
        K, N = ctx_img.shape[0], img_act.shape[0]
        if ctx_act.shape[0] != K:
            raise ValueError(f"context_actions({ctx_act.shape[0]}) 与 context_images({K}) 数量不一致")

        # 1) 后验热身：逐帧 observe，动作按到达帧索引
        state = self.rssm.initial_state(1, device=self.device)
        ctx_feats = []
        for t in range(K):
            state, _ = self.rssm.observe(state, ctx_act[t], self._encode(ctx_img[t]))
            ctx_feats.append(self.rssm.get_feature(state).reshape(-1))   # (F,)

        # 2) 开环想象：纯动力学展开 N 步
        states, _ = self.rssm.imagine_rollout(state, img_act, sample=sample)

        # 3) 批量解码 + 头预测（本接口为单轨迹推理，batch 维固定为 1）
        imgs = self.rssm.decode_rollout(states, self.decoder).clamp(-1, 1)   # (N,1,3,S,S)
        feats = torch.stack([self.rssm.get_feature(s) for s in states], 0)   # (N,1,F)
        feats = feats.reshape(feats.shape[0], -1)                            # (N,F)
        rewards = self.reward_head(feats).squeeze(-1) if self.reward_head is not None else None
        values = self.value_head(feats).squeeze(-1) if self.value_head is not None else None

        return {"images": imgs[:, 0], "states": states, "features": feats,
                "rewards": rewards, "values": values,
                "context_features": torch.stack(ctx_feats, 0)}

    @torch.no_grad()
    def rollout_from_trajectory(self, images: torch.Tensor, actions: torch.Tensor,
                                warmup_steps: int = 1, horizon: int = 10,
                                start: int = 0, sample: bool = True) -> Dict:
        """从整条轨迹切片的便捷入口。

        参数：
          images:  (T,3,S,S) 归一化 [-1,1] 帧序列
          actions: (T,4)     归一化 [-1,1] 动作序列（按到达帧索引）
          warmup_steps: 后验热身帧数 K（使用帧 start..start+K-1）
          horizon:      开环想象步数 N（预测帧 start+K .. start+K+N-1）
          start:        起始帧索引
        """
        K, N = warmup_steps, horizon
        need = start + K + N
        if images.shape[0] < need or actions.shape[0] < need:
            raise ValueError(f"轨迹长度不足：需要 {need} 帧，实际 {images.shape[0]}")
        ctx_img = images[start:start + K]
        ctx_act = actions[start:start + K]
        img_act = actions[start + K:start + K + N]
        return self.rollout(ctx_img, ctx_act, img_act, sample=sample)


if __name__ == "__main__":
    # 自检：随机权重下跑通全链路
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = ConvEncoder(latent_dim=64).to(dev)
    dec = ConvDecoder(latent_dim=264, out_size=64).to(dev)
    rssm = RSSM(stoch_dim=64, det_dim=200, action_dim=4, embed_dim=64).to(dev)
    wm = WorldModel(enc, dec, rssm, RewardHead(264).to(dev), ValueHead(264).to(dev),
                    64, 64, 200, 64, torch.device(dev))
    imgs = torch.randn(15, 3, 64, 64)
    acts = torch.rand(15, 4) * 2 - 1
    out = wm.rollout_from_trajectory(imgs, acts, warmup_steps=3, horizon=10)
    assert out["images"].shape == (10, 3, 64, 64), out["images"].shape
    assert out["features"].shape == (10, 264)
    assert out["rewards"].shape == (10,) and out["values"].shape == (10,)
    assert len(out["states"]) == 10
    assert out["context_features"].shape == (3, 264)
    print(f"SELF_TEST_OK images={tuple(out['images'].shape)} "
          f"features={tuple(out['features'].shape)} rewards={tuple(out['rewards'].shape)}")
