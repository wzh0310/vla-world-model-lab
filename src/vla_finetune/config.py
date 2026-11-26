"""微调超参数配置：dataclass + YAML 加载 + 命令行覆盖。

设计原则：
- 所有与动作空间 / 统计量相关的口径集中在此，避免训练、推理、环境三处漂移。
- YAML 支持平铺键与分节键（分节会被展开为平铺），便于按主题组织。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class LoRAFinetuneConfig:
    # ---- 路径 ----
    model_dir: str = "/data/wzh/robot/models/openvla-7b"
    train_npz: str = "/data/wzh/robot/datasets/vla_pushcube_25/v1.0/train_v1.npz"
    val_npz: str = "/data/wzh/robot/datasets/vla_pushcube_25/v1.0/val_v1.npz"
    norm_stats: str = "/data/wzh/robot/datasets/vla_pushcube_25/v1.0/norm_stats.json"
    out_dir: str = "/data/wzh/robot/vla-world-model-lab/results/vla_finetune"
    run_name: str = "pushcube_lora_v1"

    # ---- 动作空间（严格对齐）----
    action_dim: int = 7          # OpenVLA 7 维：[dx,dy,dz,drx,dry,drz,gripper]
    source_action_dim: int = 4   # 数据集 4 维：[dx,dy,dz,gripper]
    unnorm_key: str = "pushcube_vla"  # 注册到模型 norm_stats 的本项目 key

    # ---- LoRA ----
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # ---- 优化 ----
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    epochs: int = 3
    max_steps: int = -1          # -1 表示按 epochs 跑满
    batch_size: int = 2
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"   # constant | cosine
    warmup_steps: int = 0

    # ---- 训练技巧 ----
    gradient_checkpointing: bool = True
    seed: int = 20261126
    device: str = "cuda:0"
    dtype: str = "bfloat16"      # 全量 LoRA 不量化（A6000 48G 足够）
    num_workers: int = 0

    # ---- 日志 / 评估 / 保存 ----
    log_every: int = 10
    eval_every: int = 100
    save_steps: int = -1
    val_max_batches: int = -1    # -1 表示全量验证集

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoRAFinetuneConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置文件顶层必须是字典: {p}")
        flat: dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(value)   # 分节展开
            else:
                flat[key] = value
        known = {f.name for f in fields(cls)}
        unknown = set(flat) - known
        if unknown:
            raise ValueError(f"配置含未知字段: {sorted(unknown)}")
        return cls(**flat)

    def apply_args(self, args: argparse.Namespace) -> "LoRAFinetuneConfig":
        """命令行显式给出的字段覆盖 YAML；未给出的（None）保留 YAML 值。"""
        for key, value in vars(args).items():
            if value is None or key in {"config", "smoke"}:
                continue
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
