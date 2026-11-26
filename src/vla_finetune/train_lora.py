#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenVLA LoRA 微调入口。

用法（容器 wzh-ai，vla 环境，仓库根目录执行）：
    # 贯通验证：加载模型 + 真实数据一个训练步 + 一个验证步 + 保存 adapter
    /opt/miniconda3/envs/vla/bin/python src/vla_finetune/train_lora.py \
        --config vla/configs/lora_finetune.yaml --smoke

    # 完整训练
    /opt/miniconda3/envs/vla/bin/python src/vla_finetune/train_lora.py \
        --config vla/configs/lora_finetune.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import Subset

from vla.data.pushcube_dataset import PushCubeFrameDataset
from vla_finetune.config import LoRAFinetuneConfig
from vla_finetune.data import ActionTokenizer
from vla_finetune.model import (
    count_trainable_params,
    load_model_and_processor,
    prepare_for_lora,
    register_custom_norm_stats,
)
from vla_finetune.trainer import LoRATrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenVLA LoRA 微调（PushCube）")
    parser.add_argument("--config", default="vla/configs/lora_finetune.yaml")
    parser.add_argument("--smoke", action="store_true", help="只跑一个训练步验证管线贯通")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-grad-ckpt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LoRAFinetuneConfig.from_yaml(ROOT / args.config).apply_args(args)
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False
    set_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[cfg] {json.dumps(cfg.to_dict(), ensure_ascii=False)}", flush=True)

    for path in (cfg.model_dir, cfg.train_npz, cfg.val_npz, cfg.norm_stats):
        if not Path(path).exists():
            raise FileNotFoundError(f"必需路径不存在: {path}")

    with open(cfg.norm_stats, encoding="utf-8") as f:
        norm_stats = json.load(f)

    train_ds = PushCubeFrameDataset(
        cfg.train_npz, norm_stats_path=cfg.norm_stats, split="train",
        with_prompt=True, return_views=False,
    )
    val_ds = PushCubeFrameDataset(
        cfg.val_npz, norm_stats_path=cfg.norm_stats, split="val",
        with_prompt=True, return_views=False,
    )
    print(f"[data] train={len(train_ds)} frames, val={len(val_ds)} frames", flush=True)

    if args.smoke:
        n_train = min(cfg.batch_size * 2, len(train_ds))
        n_val = min(max(cfg.batch_size, 1), len(val_ds))
        train_ds = Subset(train_ds, range(n_train))
        val_ds = Subset(val_ds, range(n_val))
        print(f"[smoke] subset train={n_train} val={n_val}", flush=True)

    t0 = time.time()
    model, processor = load_model_and_processor(cfg)
    print(
        f"[model] loaded in {time.time()-t0:.1f}s dtype={cfg.dtype} "
        f"vocab_size={getattr(model, 'vocab_size', '?')} "
        f"n_bins={getattr(model.config, 'n_action_bins', '?')}",
        flush=True,
    )

    entry = register_custom_norm_stats(model, norm_stats, cfg.unnorm_key)
    print(f"[norm] registered unnorm_key={cfg.unnorm_key} q01={entry['q01']} q99={entry['q99']}", flush=True)

    tokenizer = ActionTokenizer(
        n_action_bins=int(model.config.n_action_bins),
        vocab_size=int(model.vocab_size),
        action_dim=cfg.action_dim,
    )
    # 动作 token 编解码自检（不依赖模型权重）
    probe = np.array([[0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0]])
    reconstruction_error = float(np.abs(tokenizer.decode(tokenizer.encode(probe)) - probe).max())
    print(f"[tokenizer] roundtrip max_err={reconstruction_error:.6f}", flush=True)

    model = prepare_for_lora(model, cfg)
    print(f"[lora] {count_trainable_params(model)}", flush=True)

    trainer = LoRATrainer(model, processor, cfg, tokenizer, train_ds, val_ds)

    if args.smoke:
        batch = next(iter(trainer.train_loader))
        train_metrics = trainer.train_step(batch, force_step=True)
        print(f"[smoke][train] {train_metrics}", flush=True)
        val_metrics = trainer.validate(max_batches=1)
        print(f"[smoke][val] {val_metrics}", flush=True)
        adapter_path = trainer.save_adapter(cfg.out_dir, step=0)
        result = {
            "mode": "smoke",
            "train": train_metrics,
            "val": val_metrics,
            "tokenizer_roundtrip_error": reconstruction_error,
            "adapter": adapter_path,
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3)
            if torch.cuda.is_available() else None,
            "elapsed_s": round(time.time() - t0, 2),
        }
        out_file = Path(cfg.out_dir) / "smoke_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[smoke] result -> {out_file}", flush=True)
        print(f"[smoke] DONE {json.dumps(result, ensure_ascii=False)}", flush=True)
    else:
        result = trainer.train()
        out_file = Path(cfg.out_dir) / "train_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[train] DONE {json.dumps(result, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
