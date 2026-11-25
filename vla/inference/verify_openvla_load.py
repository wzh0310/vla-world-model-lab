#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最简 OpenVLA 加载验证：处理器 + 权重 + 可选单步动作推理。

用法（vla 环境）：
    python scripts/verify_openvla_load.py --model-dir /data/wzh/robot/models/openvla-7b
    python scripts/verify_openvla_load.py --model-dir ... --predict --quant int4
"""

import argparse
import importlib.metadata as md
import os
import sys
import time


def print_versions():
    print("[环境] Python", sys.version.split()[0])
    for pkg in ("torch", "transformers", "tokenizers", "timm", "accelerate", "peft",
                "pillow", "huggingface-hub", "numpy"):
        try:
            print(f"[环境] {pkg}={md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"[环境] {pkg}=未安装")


def build_args():
    p = argparse.ArgumentParser(description="验证 OpenVLA 权重与处理器可正常初始化")
    p.add_argument("--model-dir", required=True, help="本地 openvla-7b 目录")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--quant", choices=("none", "int4"), default="none",
                   help="none=bf16；int4=nf4 4bit 量化（低显存）")
    p.add_argument("--predict", action="store_true", help="额外执行一次最小动作推理")
    p.add_argument("--unnorm-key", default=None, help="predict 时的反归一化统计量键")
    return p.parse_args()


def main():
    args = build_args()
    print_versions()

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    if not os.path.isdir(args.model_dir):
        raise SystemExit(f"模型目录不存在: {args.model_dir}")
    need = ("config.json", "model.safetensors.index.json", "processing_prismatic.py")
    for f in need:
        if not os.path.exists(os.path.join(args.model_dir, f)):
            raise SystemExit(f"缺少关键文件: {f}")
    print(f"[模型] 目录={args.model_dir}")
    print(f"[模型] 文件数={len(os.listdir(args.model_dir))} "
          f"体积={sum(os.path.getsize(os.path.join(args.model_dir, f)) for f in os.listdir(args.model_dir)) / 1e9:.2f} GB")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    print(f"[处理器] 初始化成功，耗时 {time.time() - t0:.1f}s，类型={type(processor).__name__}")

    kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    if args.quant == "int4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kw.pop("torch_dtype")
        # 量化模型由 bitsandbytes 直接落到设备，禁止再调用 .to()
        kw["device_map"] = {"": args.device.split(":")[0] or 0}
    else:
        kw["attn_implementation"] = "sdpa"

    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(args.model_dir, **kw)
    print(f"[模型] 权重加载成功，耗时 {time.time() - t0:.1f}s")
    if args.quant == "none":
        model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数量={n_params / 1e9:.2f}B， dtype={next(model.parameters()).dtype}")
    if torch.cuda.is_available():
        print(f"[显存] 已分配={torch.cuda.memory_allocated() / 1e9:.2f} GB，"
              f"保留={torch.cuda.memory_reserved() / 1e9:.2f} GB")

    norm_stats = getattr(model.config, "norm_stats", None) or {}
    print(f"[模型] 反归一化统计量键={list(norm_stats.keys())[:8]}")

    if args.predict:
        image = Image.fromarray(
            (torch.rand(224, 224, 3) * 255).to(torch.uint8).numpy()
        )
        prompt = ("In: What action should the robot take to push the cube to the target?\nOut:")
        # 4bit 量化时视觉编码器权重为 fp16，输入需同 dtype
        in_dtype = torch.float16 if args.quant == "int4" else torch.bfloat16
        inputs = processor(prompt, image).to(args.device, dtype=in_dtype)
        key = args.unnorm_key or (next(iter(norm_stats)) if norm_stats else None)
        print(f"[推理] unnorm_key={key}")
        t0 = time.time()
        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
        print(f"[推理] 成功，耗时 {time.time() - t0:.2f}s，动作 shape={tuple(action.shape)}，"
              f"值={[round(float(v), 4) for v in action.flatten()[:7]]}")

    print("[验证] 全部通过")


if __name__ == "__main__":
    main()
