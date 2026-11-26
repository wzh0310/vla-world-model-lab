#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 HF Dataset 适配器 + FinetuneCollator 能产出合法 batch（vla 环境，不加载 7B）。

检查适配器输出字段、图像/动作形状；用 mock processor 跑通 collator，确认
input_ids/labels/pixel_values 形状与 action token 监督位正确。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class MockProcessor:
    def __call__(self, text, images, return_tensors="pt"):
        n = len(text)
        ids = torch.randint(100, 1000, (n, 16))
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(n, 16, dtype=torch.long),
            "pixel_values": torch.zeros(n, 6, 224, 224),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset")
    ap.add_argument("--repo", default="/data/wzh/robot/vla-world-model-lab")
    args = ap.parse_args()

    checks = []

    def ck(name, cond, detail=""):
        checks.append({"name": name, "ok": bool(cond), "detail": str(detail)})

    ad = _load(os.path.join(args.repo, "vla_finetune", "data", "hf_dataset_adapter.py"), "_ad")
    ft_data = _load(os.path.join(args.repo, "src", "vla_finetune", "data.py"), "_ftd")

    ds = ad.HFPushCubeDataset(os.path.join(args.data, "train"))
    s0 = ds[0]
    ck("image_shape", tuple(s0["image"].shape) == (3, 224, 224), tuple(s0["image"].shape))
    ck("action_shape", tuple(s0["action"].shape) == (4,), tuple(s0["action"].shape))
    ck("prompt_format", s0["prompt"].startswith("In: What action") and s0["prompt"].endswith("Out:"),
       s0["prompt"][:40])

    tok = ft_data.ActionTokenizer(n_action_bins=256, vocab_size=32000, action_dim=7)
    coll = ft_data.FinetuneCollator(MockProcessor(), tok, action_dim=7)
    batch = coll([ds[i] for i in range(3)])
    ck("input_ids_shape", tuple(batch["input_ids"].shape) == (3, 23),
       tuple(batch["input_ids"].shape))  # 16 prompt + 7 action
    ck("pixel_values_shape", tuple(batch["pixel_values"].shape) == (3, 6, 224, 224),
       tuple(batch["pixel_values"].shape))
    labels = batch["labels"]
    ck("labels_prompt_ignored", bool((labels[:, :16] == ft_data.IGNORE_INDEX).all()))
    ck("labels_action_supervised", bool((labels[:, 16:] != ft_data.IGNORE_INDEX).all()))

    ok = all(c["ok"] for c in checks)
    result = {"ok": ok, "passed": sum(c["ok"] for c in checks),
              "total": len(checks), "checks": checks}
    out = os.path.join(args.data, "adapter_verify_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[适配器校验] {result['passed']}/{result['total']} ok={ok}")
    for c in checks:
        if not c["ok"]:
            print("  FAIL", c["name"], c["detail"])
    print(f"[结果] {out}")


if __name__ == "__main__":
    main()
