# -*- coding: utf-8 -*-
"""把跨任务示教 npz 组装成 few-shot HF Dataset（vla 环境执行）。

train = 5 条新示教（归一化 4D 动作），val = 原任务 val 子集（监控灾难性遗忘）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value, Sequence, load_from_disk


def normalize_actions(acts, norm):
    lo = np.asarray(norm["min"], dtype=np.float64)
    hi = np.asarray(norm["max"], dtype=np.float64)
    span = np.maximum(hi - lo, 1e-8)
    out = np.zeros_like(acts, dtype=np.float32)
    for j in range(4):
        if span[j] > 1e-8:
            out[:, j] = np.clip(2.0 * (acts[:, j] - lo[j]) / span[j] - 1.0, -1, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="results/vla_cross_task")
    ap.add_argument("--norm-stats", default="/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset/norm_stats.json")
    args = ap.parse_args()
    base = Path(args.base)
    norm = json.load(open(args.norm_stats, encoding="utf-8"))["action"]

    d = np.load(base / "cross_demos_raw.npz", allow_pickle=True)
    imgs = d["images"]; acts = d["actions"]; instrs = d["instructions"]
    acts_norm = normalize_actions(acts, norm)
    features = Features({
        "image": HFImage(), "instruction": Value("string"),
        "action": Sequence(Value("float32"), length=4),
    })
    ds_train = Dataset.from_dict({
        "image": [Image.fromarray(x) for x in imgs],
        "instruction": [str(x) for x in instrs],
        "action": acts_norm.tolist(),
    }, features=features)
    ds_train.save_to_disk(str(base / "raw_cross" / "train"))

    main_val = load_from_disk("/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset/val")
    n_val = min(64, len(main_val))
    ds_val = main_val.select(range(n_val))
    ds_val.save_to_disk(str(base / "raw_cross" / "val"))

    meta_path = base / "cross_demos_meta.json"
    meta = json.load(open(meta_path, encoding="utf-8"))
    meta["val_note"] = f"原任务 val 前 {n_val} 帧，监控灾难性遗忘；新任务数据不进 val"
    meta["train_frames"] = len(ds_train)
    meta["val_frames"] = n_val
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    from collections import Counter
    print("[build] train n=", len(ds_train), "val n=", n_val)
    print("[build] instr:", dict(Counter(ds_train["instruction"])))
    a = np.asarray(ds_train["action"])
    print("[build] action min/max:", a.min(axis=0).round(3), a.max(axis=0).round(3))
    print("BUILD_CROSS_DONE")


if __name__ == "__main__":
    main()
