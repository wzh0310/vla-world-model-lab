#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""动作归一化 / 反归一化工具（训练与推理共用同一口径）。

数据集 v1.0 的 action 已按 min-max 映射到 [-1,1]（4 维 [dx,dy,dz,gripper]），
OpenVLA 词表对应 7 维 [dx,dy,dz,drx,dry,drz,gripper]。本模块把口径收敛到一处：

1. normalize(physical) / denormalize(unit)：min-max <-> [-1,1]，常量维单独处理；
2. expand_4d_to_7d / squeeze_7d_to_4d：4D 与 7D 互转，旋转三维置 0；
3. ActionTokenizer：[-1,1] 连续动作 <-> OpenVLA 离散动作 token（与 predict_action 互逆）。

训练侧把归一化 4 维扩成归一化 7 维再 token 化；推理侧把 7 维动作 token
解码后反归一化到物理 4 维，交给 MuJoCo。两侧必须共用同一个 norm_stats。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActionNorm:
    """min-max <-> [-1,1] 的动作尺度转换器，常量维（span≈0）直接写 0。"""

    def __init__(self, min_values, max_values, eps: float = 1e-8):
        self.lo = np.asarray(min_values, dtype=np.float64)
        self.hi = np.asarray(max_values, dtype=np.float64)
        if self.lo.shape != self.hi.shape:
            raise ValueError(f"min/max 维度不一致: {self.lo.shape} vs {self.hi.shape}")
        self.span = self.hi - self.lo
        self.valid = self.span > eps
        self.eps = eps

    @classmethod
    def from_json(cls, path: str | Path) -> "ActionNorm":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        action = d["action"] if "action" in d else d
        return cls(action["min"], action["max"])

    def normalize(self, physical):
        """物理动作 -> [-1,1]；常量维置 0。"""
        a = np.asarray(physical, dtype=np.float64)
        unit = np.zeros_like(a)
        unit[..., self.valid] = (
            2.0 * (a[..., self.valid] - self.lo[self.valid])
            / np.maximum(self.span[self.valid], self.eps) - 1.0
        )
        return np.clip(unit, -1.0, 1.0)

    def denormalize(self, unit):
        """[-1,1] -> 物理动作；常量维置其 min 值。"""
        u = np.clip(np.asarray(unit, dtype=np.float64), -1.0, 1.0)
        phys = np.broadcast_to(self.lo, u.shape).astype(np.float64).copy()
        phys[..., self.valid] = (
            self.lo[self.valid]
            + (u[..., self.valid] + 1.0) * self.span[self.valid] / 2.0
        )
        return phys


def expand_4d_to_7d(action):
    """[dx,dy,dz,gripper] -> [dx,dy,dz,0,0,0,gripper]（归一化域或物理域均可用）。"""
    a = np.asarray(action, dtype=np.float64)
    if a.shape[-1] == 7:
        return a
    if a.shape[-1] != 4:
        raise ValueError(f"期望 4 维或 7 维动作，实际 {a.shape}")
    out = np.zeros(a.shape[:-1] + (7,), dtype=np.float64)
    out[..., :3] = a[..., :3]
    out[..., 6] = a[..., 3]
    return out


def squeeze_7d_to_4d(action):
    """[dx,dy,dz,drx,dry,drz,gripper] -> [dx,dy,dz,gripper]（丢弃旋转）。"""
    a = np.asarray(action, dtype=np.float64)
    if a.shape[-1] == 4:
        return a
    if a.shape[-1] != 7:
        raise ValueError(f"期望 4 维或 7 维动作，实际 {a.shape}")
    out = np.zeros(a.shape[:-1] + (4,), dtype=np.float64)
    out[..., :3] = a[..., :3]
    out[..., 3] = a[..., 6]
    return out


class ActionTokenizer:
    """连续动作 [-1,1] <-> OpenVLA 离散动作 token id。

    OpenVLA 把每维动作离散为 n_action_bins 个 bin，token id 取词表尾部：
        token_id = vocab_size - 1 - bin_index
    与 modeling_prismatic.py 的 predict_action 解码逻辑互逆。
    """

    def __init__(self, n_action_bins: int, vocab_size: int, action_dim: int = 7):
        self.n_action_bins = int(n_action_bins)
        self.vocab_size = int(vocab_size)
        self.action_dim = int(action_dim)
        bins = np.linspace(-1.0, 1.0, self.n_action_bins)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0

    def encode(self, actions):
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        if a.shape[-1] != self.action_dim:
            raise ValueError(f"动作维度 {a.shape[-1]} != {self.action_dim}")
        idx = np.abs(self.bin_centers[None, None, :] - a[:, :, None]).argmin(axis=-1)
        idx = np.clip(idx, 0, len(self.bin_centers) - 1)
        return (self.vocab_size - 1 - idx).astype(np.int64)

    def decode(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        disc = np.clip(self.vocab_size - 1 - ids, 0, len(self.bin_centers) - 1)
        return self.bin_centers[disc]


def roundtrip_error(tokenizer: ActionTokenizer, actions) -> float:
    """token 编解码最大绝对误差，用于管线自检。"""
    a = np.asarray(actions, dtype=np.float64)
    return float(np.abs(tokenizer.decode(tokenizer.encode(a)) - a).max())
