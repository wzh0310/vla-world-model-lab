"""模型加载、LoRA 挂载与本项目 unnorm_key 注册。

LoRA 策略（与 docs/LoRA微调配置与数据集字段规范.md 一致）：
- 只挂 LLM 主干（Llama-2）的全部线性层 q/k/v/o + gate/up/down；
- 视觉编码器 DINOv2 + SigLIP 与 projector 全部冻结；
- 不做量化（A6000 48GB 足够；QLoRA 会掉精度）。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor

# 只匹配语言主干；不匹配 vision_backbone.* / projector.*（视觉塔冻结）。
# 注意：peft==0.11.1 仅当 target_modules 为单个字符串时才按正则 fullmatch；
# 传 list 会退化为精确名/后缀匹配（正则不会生效），故合并为一条正则。
LLM_LINEAR_PATTERN = (
    r"language_model\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def load_model_and_processor(cfg):
    """用 trust_remote_code 加载 OpenVLA（AutoModelForVision2Seq）与 PrismaticProcessor。"""
    dtype = getattr(torch, cfg.dtype)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(cfg.model_dir, trust_remote_code=True)
    return model, processor


def register_custom_norm_stats(model, norm_stats: dict, key: str) -> dict:
    """把本项目 min-max 统计量注册为模型 unnorm_key，供微调后推理反归一化。

    OpenVLA predict_action 使用 0.5*(a+1)*(q99-q01)+q01 反归一化；本项目
    min-max 反演为 (a+1)*(max-min)/2+min，令 q01=min、q99=max 即可完全等价。
    4D -> 7D 扩维时旋转三维置 0 且 mask=False（常量维不参与反归一化）。
    """
    action = norm_stats["action"]
    lo = [float(x) for x in action["min"]]
    hi = [float(x) for x in action["max"]]
    if len(lo) != 4 or len(hi) != 4:
        raise ValueError(f"期望 4 维 norm_stats，实际 min={len(lo)} max={len(hi)}")
    q01 = lo[:3] + [0.0, 0.0, 0.0] + [lo[3]]
    q99 = hi[:3] + [0.0, 0.0, 0.0] + [hi[3]]
    mask = [True, True, True, False, False, False, abs(hi[3] - lo[3]) > 1e-8]
    entry = {"q01": q01, "q99": q99, "mask": mask}
    # 在包裹 LoRA 之前写入底座模型；PEFT 会保留该属性。
    if not hasattr(model, "norm_stats") or model.norm_stats is None:
        model.norm_stats = {}
    model.norm_stats[key] = entry
    return entry


def prepare_for_lora(model, cfg):
    """冻结底座 + 挂 LoRA + 梯度检查点 + 搬设备。"""
    for param in model.parameters():
        param.requires_grad_(False)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LLM_LINEAR_PATTERN,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if hasattr(model, "config"):
        model.config.use_cache = False

    if cfg.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        # 梯度检查点要求段输入 requires_grad；在底座词嵌入上挂 hook。
        base = model.base_model.model if isinstance(model, PeftModel) else model
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    return model.to(cfg.device)


def count_trainable_params(model) -> dict:
    """统计可训练 / 总参数量，用于确认 LoRA 只训练少量参数。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": round(100.0 * trainable / max(total, 1), 4),
    }


def save_inference_meta(model, cfg, out_dir: str | Path) -> Path:
    """落盘推理所需元信息：unnorm_key、动作维度、词表口径、统计量副本。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = model.base_model.model if isinstance(model, PeftModel) else model
    entry = None
    if getattr(base, "norm_stats", None) and cfg.unnorm_key in base.norm_stats:
        entry = base.norm_stats[cfg.unnorm_key]
    meta = {
        "model_dir": cfg.model_dir,
        "unnorm_key": cfg.unnorm_key,
        "action_dim": cfg.action_dim,
        "source_action_dim": cfg.source_action_dim,
        "n_action_bins": int(getattr(base.config, "n_action_bins", 256)),
        "vocab_size": int(getattr(base, "vocab_size", 32000)),
        "norm_stats_entry": entry,
        "lora": {
            "rank": cfg.lora_rank,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "target_modules": LLM_LINEAR_PATTERN,
        },
    }
    with open(out / "finetune_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out / "finetune_meta.json"
