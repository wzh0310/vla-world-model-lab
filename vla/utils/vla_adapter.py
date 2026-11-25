#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenVLA 与 PushCube 场景之间的通用适配工具。

把闭环推理反复用到的能力收敛到一处，供闭环脚本与诊断脚本复用：

1. 图像预处理：环境 RGB 观测 → 224×224 模型输入，带通道顺序校验与 BGR 对照开关；
2. 动作映射：OpenVLA 反归一化的 7 维物理增量 → PushCube 4 维动作（丢弃旋转）；
3. 结果解析：把动作向量解析为带语义的可读字典；
4. 模型封装：OpenVLAPolicy 统一 bf16 / int4 两种加载与单步推理。

本模块只依赖 vla 环境的包（torch / PIL / numpy / transformers），
不依赖 gymnasium 或 mujoco，便于在进程间复用。
"""

import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# ---- 常量 ------------------------------------------------------------------
ACTION_NAMES_7D = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
ACTION_NAMES_4D = ("dx", "dy", "dz", "gripper")
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"
DEFAULT_MODEL_DIR = "/data/wzh/robot/models/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"


def build_prompt(instruction, template=PROMPT_TEMPLATE):
    """按官方模板构造推理 prompt。"""
    return template.format(instruction=instruction.strip())


def preprocess_image(image, size=224, channel_mode="rgb"):
    """把环境观测转成模型输入用的 PIL 图像。

    参数
    ----
    image : np.ndarray | PIL.Image | str
        环境返回的 RGB 观测、PIL 图像，或图像文件路径。
    size : int
        目标边长，OpenVLA 固定 224。
    channel_mode : {"rgb", "bgr"}
        "rgb" 保持原通道（正常路径）；"bgr" 交换 R/B 通道，
        仅用于诊断"通道顺序错误"这一常见问题时的对照实验。

    返回
    ----
    PIL.Image.Image : 已转 RGB、已 resize 的图像。
    dict : 预处理元信息（原始尺寸、dtype、通道模式）。
    """
    meta = {"channel_mode": channel_mode}
    if isinstance(image, str):
        pil = Image.open(image)
        meta["source"] = "path"
    elif isinstance(image, Image.Image):
        pil = image
        meta["source"] = "pil"
    else:
        arr = np.asarray(image)
        meta["source"] = "ndarray"
        meta["raw_shape"] = list(arr.shape)
        meta["raw_dtype"] = str(arr.dtype)
        # 环境 obs["image"] 为 uint8 RGB；float 输入先按 0-1 还原
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # ndarray 一律按 RGB 解释；BGR 对照时交换 R/B
        if channel_mode == "bgr" and arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., [2, 1, 0]].copy()
            meta["swapped_rb"] = True
        pil = Image.fromarray(arr, mode="RGB")

    pil = pil.convert("RGB")
    meta["orig_size"] = list(pil.size)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    meta["out_size"] = list(pil.size)
    return pil, meta


def unnorm_to_unit(raw_7d, norm_stats, unnorm_key):
    """把 OpenVLA 已反归一化的物理增量还原到 [-1, 1] 归一化区间。

    OpenVLA predict_action 返回的是按数据集 q01/q99 反归一化后的值；
    这里用同一统计量做逆变换，得到策略输出的原始归一化动作，
    再映射到本环境量程，避免把物理量当成归一化量二次缩放。
    """
    stats = norm_stats[unnorm_key]["action"]
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    raw = np.asarray(raw_7d, dtype=np.float64).reshape(-1)[:7]
    unit = 2.0 * (raw - q01) / np.maximum(q99 - q01, 1e-8) - 1.0
    return np.clip(unit, -1.0, 1.0)


def map_action(raw_7d, norm_stats, unnorm_key=DEFAULT_UNNORM_KEY,
               scale=0.02, flip_xyz=False):
    """OpenVLA 7 维动作 → PushCube 4 维动作。

    丢弃 3 维旋转增量（推物任务无需旋转）；位置增量按 ±scale 米线性映射；
    夹爪从 [-1,1] 线性映射到 [0,1]（本环境接口保留、不参与动力学）。

    flip_xyz=True 时把位置三维取反，用于诊断"动作映射方向反转"。
    """
    unit = unnorm_to_unit(raw_7d, norm_stats, unnorm_key)
    xyz = unit[:3].copy()
    if flip_xyz:
        xyz = -xyz
    xyz_m = (xyz * scale).astype(float)
    gripper = float((unit[6] + 1.0) / 2.0)
    mapped = [float(v) for v in xyz_m] + [gripper]
    return {
        "raw_7d": [float(v) for v in np.asarray(raw_7d).reshape(-1)[:7]],
        "normalized_7d": [float(v) for v in unit],
        "ignored_rotation": [float(v) for v in unit[3:6]],
        "mapped_4d": mapped,
        "flip_xyz": bool(flip_xyz),
        "scale_m": float(scale),
    }


def parse_action(mapped_4d, names=ACTION_NAMES_4D):
    """把 4 维动作向量解析为带语义的可读字典。"""
    a = np.asarray(mapped_4d, dtype=float).reshape(-1)[:4]
    d = {name: round(float(v), 6) for name, v in zip(names, a)}
    horiz = float(np.linalg.norm(a[:2]))
    d["horizontal_step_m"] = round(horiz, 6)
    d["move_direction"] = ("接近目标" if a[0] > 0.001 else
                           "远离目标" if a[0] < -0.001 else "横向为主")
    d["gripper_state"] = "闭合" if a[3] < 0.5 else "张开"
    return d


class OpenVLAPolicy:
    """封装 OpenVLA 加载与单步推理，统一 bf16 / int4 两种精度。"""

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, quant="none",
                 device="cuda:0", unnorm_key=DEFAULT_UNNORM_KEY):
        self.model_dir = model_dir
        self.quant = quant
        self.device = device
        self.unnorm_key = unnorm_key
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

        kw = dict(low_cpu_mem_usage=True, trust_remote_code=True)
        if quant == "int4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            # 量化模型由 bitsandbytes 直接落设备，禁止再 .to()
            kw["device_map"] = {"": device.split(":")[0] or 0}
        else:
            kw["torch_dtype"] = torch.bfloat16
            kw["attn_implementation"] = "sdpa"
        self.model = AutoModelForVision2Seq.from_pretrained(model_dir, **kw)
        if quant == "none":
            self.model = self.model.to(device)
        self.model.eval()
        self.norm_stats = getattr(self.model.config, "norm_stats", None) or {}
        if unnorm_key not in self.norm_stats:
            raise KeyError(f"unnorm_key={unnorm_key} 不在 norm_stats 中；"
                           f"可用键: {list(self.norm_stats.keys())[:8]}")
        # 4bit 量化时视觉编码器权重为 fp16，输入需同 dtype
        self.input_dtype = torch.float16 if quant == "int4" else torch.bfloat16
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def predict(self, image, instruction, channel_mode="rgb",
                scale=0.02, flip_xyz=False, warmup=True):
        """对单帧观测执行推理并返回映射结果。"""
        pil, prep_meta = preprocess_image(image, channel_mode=channel_mode)
        prompt = build_prompt(instruction)
        inputs = self.processor(prompt, pil).to(self.device, dtype=self.input_dtype)
        with torch.inference_mode():
            if warmup:
                self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False)
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        result = map_action(
            action, self.norm_stats, self.unnorm_key, scale=scale, flip_xyz=flip_xyz)
        result.update({
            "instruction": instruction,
            "prompt": prompt,
            "unnorm_key": self.unnorm_key,
            "latency_s": round(latency, 4),
            "preprocess": prep_meta,
            "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        })
        return result

    def memory_gb(self):
        return {
            "allocated": round(torch.cuda.memory_allocated() / 1e9, 3),
            "reserved": round(torch.cuda.memory_reserved() / 1e9, 3),
        }
