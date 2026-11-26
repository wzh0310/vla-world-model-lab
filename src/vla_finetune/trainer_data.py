#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Trainer 数据管线：HuggingFace Dataset -> OpenVLA forward 可直接消费的 batch。

严格对齐点：
1. prompt 直接调用 vla/utils/vla_adapter.py::build_prompt（官方模板，
   "In: What action should the robot take to {instruction}?\\nOut:"），
   避免训练/推理模板漂移导致微调失效；
2. 数据集 action 为 4 维 [-1,1]，经 action_utils 扩成 7 维归一化动作再 token 化；
3. labels 只对动作 token 位监督，prompt 位与图像 patch 位均置 IGNORE_INDEX。
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from .action_utils import ActionTokenizer, expand_4d_to_7d

IGNORE_INDEX = -100

# 与 vla_adapter 同源：按文件路径加载，规避 utils 包遮蔽
_ADAPTER_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "vla", "utils", "vla_adapter.py"))
_spec = importlib.util.spec_from_file_location("vla_adapter_ft", _ADAPTER_PATH)
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)
build_prompt = _adapter.build_prompt
PROMPT_TEMPLATE = _adapter.PROMPT_TEMPLATE


class HFDatasetForTrainer(Dataset):
    """把 HuggingFace Dataset split 包装成 PyTorch Dataset。

    每条样本返回 image(PIL) / instruction(str) / action(4D 归一化 ndarray)。
    """

    def __init__(self, split_dir: str):
        self.ds = load_from_disk(split_dir)
        required = {"image", "instruction", "action"}
        missing = required - set(self.ds.column_names)
        if missing:
            raise ValueError(f"{split_dir} 缺少字段: {sorted(missing)}")
        self.prompts = [build_prompt(str(x)) for x in self.ds["instruction"]]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "image": item["image"].convert("RGB"),
            "prompt": self.prompts[idx],
            "instruction": str(item["instruction"]),
            "action": np.asarray(item["action"], dtype=np.float64),
        }


class TrainerCollator:
    """把样本整理成 OpenVLA forward batch；prompt 与动作 token 严格对齐。"""

    def __init__(self, processor, action_tokenizer: ActionTokenizer, action_dim: int = 7):
        self.processor = processor
        self.tokenizer = action_tokenizer
        self.action_dim = action_dim

    def __call__(self, items):
        prompts = [x["prompt"] for x in items]
        images = [x["image"] for x in items]
        # processor 内部按 DINOv2 / SigLIP 双主干口径归一化，输出 pixel_values [B,6,224,224]
        # 不同指令 token 长度不同，必须 padding 才能组成 batch tensor。
        encoded = self.processor(text=prompts, images=images,
                                 padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        actions = np.stack([
            expand_4d_to_7d(x["action"]) for x in items
        ])  # [B,7] 归一化域
        action_tokens = torch.from_numpy(self.tokenizer.encode(actions)).long()  # [B,7]

        new_input_ids = torch.cat([input_ids, action_tokens], dim=1)
        new_attention_mask = torch.cat(
            [attention_mask, torch.ones_like(action_tokens)], dim=1)
        labels = torch.full_like(new_input_ids, IGNORE_INDEX)
        labels[:, input_ids.shape[1]:] = action_tokens

        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "pixel_values": encoded["pixel_values"],
            "labels": labels,
        }
