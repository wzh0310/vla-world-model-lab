"""训练循环、验证评估与 adapter 保存。

loss 口径：只对动作 token 位计算交叉熵（prompt 位与图像 patch 位均为 -100）。
验证指标：val_loss + action_token_accuracy（7 维动作 token 逐个比对）。
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import IGNORE_INDEX, FinetuneCollator
from .model import save_inference_meta


class LoRATrainer:
    def __init__(self, model, processor, cfg, action_tokenizer, train_dataset, val_dataset):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.action_dim = cfg.action_dim

        self.collator = FinetuneCollator(processor, action_tokenizer, cfg.action_dim)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("没有可训练参数：检查 LoRA target_modules 与冻结逻辑")
        self.params = params
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / cfg.grad_accumulation_steps))
        self.total_steps = cfg.max_steps if cfg.max_steps > 0 else cfg.epochs * steps_per_epoch
        self.scheduler = self._build_scheduler()

        self.global_step = 0
        self.micro_step = 0
        self.history: list[dict] = []

    def _build_scheduler(self):
        if self.cfg.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.total_steps)
            )

        def lr_fn(step):
            if self.cfg.warmup_steps > 0 and step < self.cfg.warmup_steps:
                return float(step + 1) / float(self.cfg.warmup_steps)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_fn)

    def _to_device(self, batch: dict) -> dict:
        """搬设备并把图像统一为模型 dtype（processor 默认输出 float32，视觉塔为 bf16）。"""
        dtype = getattr(torch, self.cfg.dtype)
        out = {}
        for key, value in batch.items():
            value = value.to(self.cfg.device, non_blocking=True)
            if key == "pixel_values" and value.is_floating_point():
                value = value.to(dtype)
            out[key] = value
        return out

    def train_step(self, batch: dict, force_step: bool = False) -> dict:
        """单个 micro-batch：前向 -> 反向 -> （按累积步）更新。"""
        self.model.train()
        batch = self._to_device(batch)
        output = self.model(**batch)
        loss = output.loss

        (loss / self.cfg.grad_accumulation_steps).backward()
        self.micro_step += 1

        grad_norm = None
        if force_step or self.micro_step % self.cfg.grad_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.params, self.cfg.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        metrics = {
            "loss": float(loss.detach().float().cpu()),
            "grad_norm": float(grad_norm) if grad_norm is not None else None,
            "lr": float(self.scheduler.get_last_lr()[0]),
            "step": self.global_step,
        }
        self.history.append(metrics)
        return metrics

    @torch.no_grad()
    def validate(self, max_batches: int | None = None) -> dict:
        """验证：平均 loss 与 7 维动作 token 逐位准确率。"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        token_count = 0
        batches = 0
        for idx, batch in enumerate(self.val_loader):
            if max_batches is not None and idx >= max_batches:
                break
            batch = self._to_device(batch)
            output = self.model(**batch)
            total_loss += float(output.loss.detach().float().cpu())
            batches += 1

            logits = output.logits[:, -self.action_dim:, :]
            labels = batch["labels"][:, -self.action_dim:]
            pred = logits.argmax(dim=-1)
            mask = labels != IGNORE_INDEX
            correct += int((pred[mask] == labels[mask]).sum().cpu())
            token_count += int(mask.sum().cpu())

        return {
            "val_loss": total_loss / max(batches, 1),
            "token_accuracy": correct / max(token_count, 1),
            "batches": batches,
            "tokens": token_count,
        }

    def save_adapter(self, out_dir: str | Path, step: int | None = None) -> str:
        """保存 LoRA adapter + processor + 推理元信息（不保存 7B 底座）。"""
        out = Path(out_dir)
        target = out / (f"checkpoint-{step}" if step is not None else "adapter")
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target))
        self.processor.save_pretrained(str(target))
        save_inference_meta(self.model, self.cfg, target)
        return str(target)

    def train(self) -> dict:
        """完整训练：按 epoch 迭代，按步记录 / 验证 / 保存。"""
        started = time.time()
        last_val: dict = {}
        for epoch in range(self.cfg.epochs):
            for batch in self.train_loader:
                if self.global_step >= self.total_steps:
                    break
                metrics = self.train_step(batch)
                if self.cfg.log_every > 0 and self.global_step % self.cfg.log_every == 0:
                    print(
                        f"[train] epoch={epoch} step={self.global_step}/{self.total_steps} "
                        f"loss={metrics['loss']:.5f} grad_norm={metrics['grad_norm']} "
                        f"lr={metrics['lr']:.2e}",
                        flush=True,
                    )
                if (
                    self.cfg.eval_every > 0
                    and self.global_step % self.cfg.eval_every == 0
                ):
                    last_val = self.validate(max_batches=self.cfg.val_max_batches
                                             if self.cfg.val_max_batches > 0 else None)
                    print(f"[eval] step={self.global_step} {last_val}", flush=True)
                if (
                    self.cfg.save_steps > 0
                    and self.global_step % self.cfg.save_steps == 0
                ):
                    path = self.save_adapter(self.cfg.out_dir, self.global_step)
                    print(f"[save] {path}", flush=True)
            if self.global_step >= self.total_steps:
                break

        last_val = self.validate(
            max_batches=self.cfg.val_max_batches if self.cfg.val_max_batches > 0 else None
        )
        final_path = self.save_adapter(self.cfg.out_dir, self.global_step)
        return {
            "steps": self.global_step,
            "elapsed_s": round(time.time() - started, 2),
            "final_val": last_val,
            "adapter": final_path,
        }
