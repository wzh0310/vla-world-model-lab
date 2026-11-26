#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 transformers Trainer API 的 OpenVLA QLoRA 微调主脚本。

配置（vla/configs/trainer_lora.yaml）要点：
- 4bit NF4 量化加载，单卡显存约 5 GB；冻结底座全部参数；
- 仅对 language_model self-attn 的 q_proj / v_proj 挂 LoRA
  （rank=8, alpha=16, dropout=0.05）——OpenVLA 把图像 patch token 与文本 token
  拼接后统一下一层 self-attn，没有独立 cross_attn 模块，图文交叉发生于此；
- lr=2e-4, batch_size=4, grad_accum=4, epochs=10, warmup_steps=50；
- prompt 由 vla_adapter.build_prompt 生成，与推理严格同源；
- 动作经 action_utils 归一化/扩维/token 化，labels 只对动作位监督。

用法（容器 vla 环境，仓库根目录）：
    # 小样本贯通（1% 数据，观察 loss 下降）
    /opt/miniconda3/envs/vla/bin/python src/vla_finetune/train.py \
        --config vla/configs/trainer_lora.yaml --smoke --smoke-warmup 2
    # 完整训练
    /opt/miniconda3/envs/vla/bin/python src/vla_finetune/train.py \
        --config vla/configs/trainer_lora.yaml
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

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model

from vla_finetune.action_utils import ActionNorm, ActionTokenizer, roundtrip_error
from vla_finetune.trainer_data import HFDatasetForTrainer, TrainerCollator

# peft 0.11.1：target_modules 为单个字符串时才按正则 fullmatch；传 list 会退化。
# 只匹配语言主干 self-attn 的 q_proj / v_proj；视觉塔与 projector 全冻结。
LORA_TARGET_PATTERN = (
    r"language_model\.model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj)"
)


def parse_args():
    ap = argparse.ArgumentParser(description="OpenVLA QLoRA 微调（Trainer API）")
    ap.add_argument("--config", default="vla/configs/trainer_lora.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="用 1% 数据跑通训练全流程并记录每步 loss")
    ap.add_argument("--smoke-warmup", type=int, default=None,
                    help="smoke 时覆盖 warmup_steps，便于短程观察 loss 下降")
    ap.add_argument("--smoke-steps", type=int, default=20,
                    help="smoke 最大 optimizer 步数")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args()


def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    flat = {}
    for k, v in data.items():
        flat.update(v) if isinstance(v, dict) else flat.__setitem__(k, v)
    return flat


class QLoRATrainer(Trainer):
    """修正 4bit 模型的输入 dtype：视觉塔权重为 fp16，pixel_values 必须同 dtype。"""

    def __init__(self, input_dtype=torch.float16, **kwargs):
        self.vla_input_dtype = input_dtype
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = dict(inputs)
        pv = inputs.get("pixel_values")
        if pv is not None and pv.is_floating_point():
            inputs["pixel_values"] = pv.to(self.vla_input_dtype)
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss 非有限值: {loss}")
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    model_dir = cfg["model_dir"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] {json.dumps(cfg, ensure_ascii=False)}", flush=True)

    # ---- 1) 4bit 加载 ----
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    device = cfg.get("device", "cuda:0")
    device_id = 0 if device == "cuda" else int(device.split(":")[-1])
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        quantization_config=bnb,
        device_map={"": device_id},  # 4bit 模型禁止再 .to()
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    vram0 = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else None
    print(f"[model] 4bit 加载 {time.time()-t0:.1f}s 参数={n_params/1e9:.2f}B "
          f"显存保留={vram0:.2f}GB", flush=True)

    # ---- 2) 注册本项目 unnorm_key（训练/推理共用） ----
    norm_stats_path = cfg["norm_stats"]
    with open(norm_stats_path, encoding="utf-8") as f:
        norm_raw = json.load(f)
    act = norm_raw["action"] if "action" in norm_raw else norm_raw
    lo = [float(x) for x in act["min"]]
    hi = [float(x) for x in act["max"]]
    unnorm_key = cfg["unnorm_key"]
    if not hasattr(model.config, "norm_stats") or model.config.norm_stats is None:
        model.config.norm_stats = {}
    model.config.norm_stats[unnorm_key] = {
        "q01": lo[:3] + [0, 0, 0] + [lo[3]],
        "q99": hi[:3] + [0, 0, 0] + [hi[3]],
        "mask": [True, True, True, False, False, False, True],
    }
    print(f"[norm] 注册 unnorm_key={unnorm_key} min={lo} max={hi}", flush=True)

    # ---- 3) 动作工具自检 ----
    action_norm = ActionNorm(lo, hi)
    tokenizer = ActionTokenizer(
        n_action_bins=int(model.config.n_action_bins),
        vocab_size=int(model.vocab_size),
        action_dim=int(cfg["action_dim"]),
    )
    probe_unit = np.array([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -0.4]])
    tok_err = roundtrip_error(tokenizer, probe_unit)
    phys_probe = action_norm.denormalize([0.0, 0.5, -0.5, 0.0])
    norm_err = float(np.abs(action_norm.normalize(phys_probe) -
                            np.array([0.0, 0.5, -0.5, 0.0])).max())
    print(f"[action] token roundtrip err={tok_err:.6f} norm roundtrip err={norm_err:.6f}",
          flush=True)

    # ---- 4) 冻结底座 + LoRA(q/v) ----
    for p in model.parameters():
        p.requires_grad_(False)
    lora_cfg = LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=LORA_TARGET_PATTERN,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] target=q_proj/v_proj r={cfg['lora_rank']} alpha={cfg['lora_alpha']} "
          f"dropout={cfg['lora_dropout']} 可训练={trainable:,} ({trainable/n_params*100:.3f}%)",
          flush=True)

    # ---- 5) 数据集 ----
    train_ds = HFDatasetForTrainer(os.path.join(cfg["hf_dataset"], "train"))
    eval_ds = HFDatasetForTrainer(os.path.join(cfg["hf_dataset"], "val"))
    collator = TrainerCollator(processor, tokenizer, action_dim=int(cfg["action_dim"]))
    smoke_note = None
    if args.smoke:
        n = max(int(len(train_ds) * 0.01), cfg["batch_size"] * cfg["grad_accum"])
        train_ds = torch.utils.data.Subset(train_ds, range(n))
        eval_n = min(cfg["batch_size"], len(eval_ds))
        eval_ds = torch.utils.data.Subset(eval_ds, range(eval_n))
        smoke_note = f"1% 数据={n} 帧（不少于 1 个完整累积 batch）"
        print(f"[smoke] {smoke_note}, eval={eval_n}", flush=True)

    # ---- 6) TrainingArguments ----
    warmup = cfg["warmup_steps"]
    if args.smoke and args.smoke_warmup is not None:
        warmup = args.smoke_warmup
        print(f"[smoke] warmup_steps 覆盖为 {warmup}（完整训练仍为 "
              f"{cfg['warmup_steps']}）", flush=True)
    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer_ckpt"),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_steps=int(warmup),
        lr_scheduler_type=cfg.get("lr_scheduler", "linear"),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        logging_steps=1,
        save_strategy="no",
        evaluation_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=False,   # 4bit + fp16 视觉输入；关闭 autocast 避免 dtype 冲突
        fp16=False,
        seed=seed,
        max_steps=args.smoke_steps if args.smoke else -1,
    )
    trainer = QLoRATrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # ---- 7) 训练（记录每步 loss） ----
    print(f"[train] 开始 epochs={cfg['epochs']} bs={cfg['batch_size']} "
          f"accum={cfg['grad_accum']} lr={cfg['learning_rate']} warmup={warmup}", flush=True)
    started = time.time()
    result = trainer.train()
    hist = [h for h in trainer.state.log_history if "loss" in h]
    losses = [float(h["loss"]) for h in hist]
    print(f"[train] 完成 steps={trainer.state.global_step} "
          f"耗时={time.time()-started:.1f}s loss步数={len(losses)}", flush=True)
    if losses:
        print(f"[train] loss 首={losses[0]:.4f} 末={losses[-1]:.4f} "
              f"最小={min(losses):.4f}", flush=True)

    # ---- 8) 验证 loss（不更新参数） ----
    eval_metrics = trainer.evaluate()
    print(f"[eval] {eval_metrics}", flush=True)

    # ---- 9) 保存 adapter + processor + 元信息 ----
    adapter_dir = out_dir / ("smoke_adapter" if args.smoke else "adapter")
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    meta = {
        "model_dir": model_dir,
        "quant": "int4-nf4",
        "unnorm_key": unnorm_key,
        "action_dim": int(cfg["action_dim"]),
        "prompt_template": "In: What action should the robot take to {instruction}?\\nOut:",
        "lora": {"rank": cfg["lora_rank"], "alpha": cfg["lora_alpha"],
                 "dropout": cfg["lora_dropout"], "target": LORA_TARGET_PATTERN},
        "norm_stats": {"min": lo, "max": hi},
    }
    with open(adapter_dir / "finetune_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    out_rec = {
        "mode": "smoke" if args.smoke else "full",
        "smoke_note": smoke_note,
        "config": {k: cfg[k] for k in ("learning_rate", "batch_size", "grad_accum",
                                       "epochs", "warmup_steps", "lora_rank",
                                       "lora_alpha", "lora_dropout")},
        "warmup_effective": warmup,
        "n_params_b": round(n_params / 1e9, 3),
        "trainable_params": trainable,
        "vram_reserved_gb_before": round(vram0, 3) if vram0 else None,
        "vram_reserved_gb_after": round(torch.cuda.memory_reserved() / 1e9, 3)
        if torch.cuda.is_available() else None,
        "token_roundtrip_error": tok_err,
        "norm_roundtrip_error": norm_err,
        "steps": trainer.state.global_step,
        "losses": [round(x, 6) for x in losses],
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
        "elapsed_s": round(time.time() - started, 2),
        "adapter": str(adapter_dir),
    }
    out_file = out_dir / ("trainer_smoke_result.json" if args.smoke
                          else "trainer_train_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_rec, f, ensure_ascii=False, indent=2)
    print(f"[done] 结果 -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
