#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把标准化 HuggingFace Dataset 包装为 PyTorch Dataset，直接对接 LoRA 微调骨架。

FinetuneCollator（src/vla_finetune/data.py）期望单条样本包含：
    image  : FloatTensor [3,224,224]，RGB，像素 [0,1]
    prompt : str，完整 OpenVLA 输入文本
    action : FloatTensor [4]，归一化到 [-1,1]
本适配器把 HF 样本的 PIL.Image / str / list 转为上述口径，其余字段原样透传。
"""
from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_PROMPT = "In: What action should the robot take to {instruction}?\nOut:"


class HFPushCubeDataset(Dataset):
    def __init__(self, split_dir: str, with_prompt: bool = True):
        self.ds = load_from_disk(split_dir)
        self.with_prompt = with_prompt
        cols = set(self.ds.column_names)
        missing = {"image", "instruction", "action"} - cols
        if missing:
            raise ValueError(f"{split_dir} 缺字段: {sorted(missing)}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img, dtype=np.uint8), mode="RGB")
        img = img.convert("RGB").resize((224, 224))
        img_t = torch.from_numpy(
            np.ascontiguousarray(np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        )
        action_t = torch.as_tensor(np.asarray(row["action"], dtype=np.float32))
        instr = str(row["instruction"])
        item = {"image": img_t, "action": action_t, "instruction": instr}
        if self.with_prompt:
            item["prompt"] = DEFAULT_PROMPT.format(instruction=instr)
        return item


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset")
    args = ap.parse_args()
    for split in ("train", "val", "test"):
        ds = HFPushCubeDataset(f"{args.data}/{split}")
        s = ds[0]
        print(f"[{split}] len={len(ds)} image={tuple(s['image'].shape)} "
              f"action={tuple(s['action'].shape)} prompt={s['prompt'][:60]!r}")
