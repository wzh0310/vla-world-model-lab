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
    TrainerCallback,
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
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="覆盖配置学习率（震荡时降至 1e-4）")
    ap.add_argument("--max-grad-norm", type=float, default=None,
                    help="覆盖梯度裁剪阈值（默认 1.0）")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="覆盖 LoRA 秩（消融实验用）")
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="覆盖 LoRA alpha（消融实验用）")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="覆盖 optimizer 总步数（消融固定预算用）")
    ap.add_argument("--lr-scheduler", default=None,
                    help="覆盖学习率调度器（消融对照用，如 constant）")
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


class BestCheckpointCallback(TrainerCallback):
    """每个 epoch 结束时比较 eval_loss，仅保留验证 loss 最低的 adapter 权重。

    4bit 量化模型的 Trainer 内建 checkpoint 会保存 optimizer/fp32 状态且禁用
    load_best_model_at_end 时行为不稳，因此这里用回调显式保存 PEFT adapter
    （几十 MB），并同步保存 processor 与元信息，规避大权重与 dtype 问题。
    """

    def __init__(self, model, processor, out_dir: Path, meta: dict, eval_every: int = 1):
        self.model = model
        self.processor = processor
        self.out_dir = Path(out_dir)
        self.meta = dict(meta)
        self.eval_every = int(eval_every)
        self.best_eval_loss = float("inf")
        self.best_epoch = None
        self.best_step = None
        self.history = []  # [{epoch, step, eval_loss, improved}]

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return
        ev = float(metrics["eval_loss"])
        if not np.isfinite(ev):
            return
        epoch = float(state.epoch or 0)
        step = int(state.global_step or 0)
        improved = ev < self.best_eval_loss
        self.history.append({
            "epoch": round(epoch, 4), "step": step,
            "eval_loss": round(ev, 6), "improved": bool(improved),
        })
        tag = "IMPROVED" if improved else "not-better"
        print(f"[checkpoint] eval@epoch={epoch:.2f} step={step} "
              f"eval_loss={ev:.4f} best={self.best_eval_loss:.4f} [{tag}]", flush=True)
        if improved:
            self.best_eval_loss = ev
            self.best_epoch = epoch
            self.best_step = step
            best_dir = self.out_dir / "best_adapter"
            best_dir.mkdir(parents=True, exist_ok=True)
            m = model if model is not None else self.model
            m.save_pretrained(str(best_dir))
            self.processor.save_pretrained(str(best_dir))
            meta = dict(self.meta)
            meta.update({
                "kind": "best_by_eval_loss",
                "eval_loss": round(ev, 6),
                "epoch": round(epoch, 4),
                "global_step": step,
            })
            with open(best_dir / "finetune_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            # 每轮刷新一份 JSON 摘要，便于训练中实时查看
            with open(self.out_dir / "best_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[checkpoint] 已保存最优权重 -> {best_dir}", flush=True)


def main():
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
        print(f"[override] learning_rate -> {args.learning_rate}", flush=True)
    if args.max_grad_norm is not None:
        cfg["max_grad_norm"] = args.max_grad_norm
        print(f"[override] max_grad_norm -> {args.max_grad_norm}", flush=True)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.lora_rank is not None:
        cfg["lora_rank"] = args.lora_rank
        print(f"[override] lora_rank -> {args.lora_rank}", flush=True)
    if args.lora_alpha is not None:
        cfg["lora_alpha"] = args.lora_alpha
        print(f"[override] lora_alpha -> {args.lora_alpha}", flush=True)
    if args.lr_scheduler is not None:
        cfg["lr_scheduler"] = args.lr_scheduler
        print(f"[override] lr_scheduler -> {args.lr_scheduler}", flush=True)

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
    eval_note = None
    if args.smoke:
        n = max(int(len(train_ds) * 0.01), cfg["batch_size"] * cfg["grad_accum"])
        train_ds = torch.utils.data.Subset(train_ds, range(n))
        eval_n = min(cfg["batch_size"], len(eval_ds))
        eval_ds = torch.utils.data.Subset(eval_ds, range(eval_n))
        smoke_note = f"1% 数据={n} 帧（不少于 1 个完整累积 batch）"
        print(f"[smoke] {smoke_note}, eval={eval_n}", flush=True)
    else:
        # 全量验证集 882 帧逐次全评估成本高；用固定前 N 帧做收敛监控与最优权重选择。
        eval_cap = int(cfg.get("eval_subset_size", 128))
        n_val = len(eval_ds)
        if 0 < eval_cap < n_val:
            eval_ds = torch.utils.data.Subset(eval_ds, range(eval_cap))
            eval_note = f"验证监控子集=前 {eval_cap}/{n_val} 帧（非全验证集）"
            print(f"[eval] {eval_note}", flush=True)

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
        save_strategy="no",  # 最优权重由 BestCheckpointCallback 显式保存，避免 4bit 内建 ckpt 问题
        evaluation_strategy="steps" if not args.smoke else "no",
        eval_steps=int(cfg.get("eval_steps", 64)) if not args.smoke else None,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=False,   # 4bit + fp16 视觉输入；关闭 autocast 避免 dtype 冲突
        fp16=False,
        seed=seed,
        max_steps=args.max_steps if args.max_steps is not None
        else (args.smoke_steps if args.smoke else -1),
    )
    trainer = QLoRATrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    best_cb = None
    if not args.smoke:
        best_cb = BestCheckpointCallback(
            model=model, processor=processor, out_dir=out_dir,
            meta={
                "model_dir": model_dir,
                "quant": "int4-nf4",
                "unnorm_key": unnorm_key,
                "action_dim": int(cfg["action_dim"]),
                "prompt_template": "In: What action should the robot take to {instruction}?\\nOut:",
                "lora": {"rank": cfg["lora_rank"], "alpha": cfg["lora_alpha"],
                         "dropout": cfg["lora_dropout"], "target": LORA_TARGET_PATTERN},
                "norm_stats": {"min": lo, "max": hi},
                "learning_rate": float(cfg["learning_rate"]),
                "max_grad_norm": float(cfg["max_grad_norm"]),
            },
        )
        trainer.add_callback(best_cb)
        print(f"[checkpoint] 每 {cfg.get('eval_steps', 64)} 步评估，"
              f"保留 eval_loss 最低的 adapter -> {out_dir / 'best_adapter'}", flush=True)

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
        "eval_note": eval_note,
        "config": {k: cfg[k] for k in ("learning_rate", "batch_size", "grad_accum",
                                       "epochs", "warmup_steps", "lora_rank",
                                       "lora_alpha", "lora_dropout", "max_grad_norm")},
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
        "best_checkpoint": {
            "eval_loss": round(best_cb.best_eval_loss, 6) if best_cb else None,
            "epoch": round(best_cb.best_epoch, 4) if best_cb and best_cb.best_epoch else None,
            "global_step": best_cb.best_step if best_cb else None,
            "eval_history": best_cb.history if best_cb else [],
            "path": str(out_dir / "best_adapter") if best_cb else None,
        },
    }
    out_file = out_dir / ("trainer_smoke_result.json" if args.smoke
                          else "trainer_train_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_rec, f, ensure_ascii=False, indent=2)
    print(f"[done] 结果 -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
