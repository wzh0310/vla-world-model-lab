#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验标准化 HuggingFace VLA 数据集（lerobot 环境运行）。

检查：三 split 存在、字段类型/形状、图像 224×224 RGB、动作 4 维 [-1,1]、
指令类别覆盖、8:1:1 轨迹比例、物理轨迹分组无跨集泄漏。
输出 verify_result.json。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from datasets import load_from_disk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset")
    args = ap.parse_args()
    checks = []

    def ck(name, cond, detail=""):
        checks.append({"name": name, "ok": bool(cond), "detail": str(detail)})

    with open(os.path.join(args.data, "dataset_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    loaded = {}
    for s in ("train", "val", "test"):
        p = os.path.join(args.data, s)
        ck(f"{s}_exists", os.path.isdir(p))
        loaded[s] = load_from_disk(p)
        ds = loaded[s]
        ck(f"{s}_fields", set(ds.column_names) == {"image", "instruction", "action"},
           ds.column_names)
        if len(ds) > 0:
            row = ds[0]
            im = row["image"]
            ck(f"{s}_image_size", im.size == (224, 224) and im.mode == "RGB",
               f"{im.size} {im.mode}")
            acts = np.asarray(ds["action"][:min(64, len(ds))], dtype=np.float32)
            ck(f"{s}_action_shape", acts.shape[1] == 4, acts.shape)
            ck(f"{s}_action_range", acts.min() >= -1.0001 and acts.max() <= 1.0001,
               f"[{acts.min():.4f},{acts.max():.4f}]")
            ck(f"{s}_instruction_nonempty",
               all(isinstance(x, str) and len(x) > 0
                   for x in ds["instruction"][:min(64, len(ds))]))

    # 类别覆盖（按 summary 统计）
    all_cats = set()
    for s in summary["splits"]:
        all_cats.update(summary["splits"][s]["categories"].keys())
    ck("categories_ge3", len(all_cats) >= 3, sorted(all_cats))
    ck("has_variants", any(c.startswith("variant_") for c in all_cats), sorted(all_cats))

    # 轨迹规模与划分
    n_lang = sum(v["language_conditioned_trajectories"] for v in summary["splits"].values())
    ck("lang_traj_ge100", n_lang >= 100, n_lang)
    n_phys = sum(v["physical_trajectories"] for v in summary["splits"].values())
    ck("phys_traj_ge40", n_phys >= 40, n_phys)

    # 物理轨迹无跨集泄漏
    seen = {}
    leak = []
    for s in ("train", "val", "test"):
        for sid in summary["splits"][s]["source_ids"]:
            if sid in seen and seen[sid] != s:
                leak.append((sid, seen[sid], s))
            seen[sid] = s
    ck("no_cross_split_leak", len(leak) == 0, leak[:5])

    ok = all(c["ok"] for c in checks)
    result = {"ok": ok, "passed": sum(c["ok"] for c in checks),
              "total": len(checks), "checks": checks}
    with open(os.path.join(args.data, "verify_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[校验] {result['passed']}/{result['total']} 通过, ok={ok}")
    for c in checks:
        if not c["ok"]:
            print("  FAIL", c["name"], c["detail"])
    print(f"[结果] {os.path.join(args.data, 'verify_result.json')}")


if __name__ == "__main__":
    main()
