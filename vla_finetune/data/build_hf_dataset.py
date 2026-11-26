#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建标准化 HuggingFace VLA 微调数据集（lerobot 环境运行）。

数据来源：
  * 基础 25 条轨迹（raw_demos_vla.npz，物理空间 4 维动作）
  * 新采集左右目标变体各 10 条（variants/variant_{left,right}.npz）

处理：
  1. 每条物理轨迹按指令同义表述展开为多条「语言条件化轨迹」：
     基础轨迹 = 原模板指令 + 2 条 paraphrase（共 3 种）；
     变体轨迹 = instruction_variants 中 4 条同义指令。
  2. 动作统一用全量轨迹重新计算的 min/max 归一化到 [-1,1]（常量维置 0），
     独立 norm_stats.json 随数据集落盘，供本阶段微调/推理复用。
  3. 按「物理轨迹」分组做 8:1:1 分层划分（同一物理轨迹的所有指令表述
     必须落在同一 split，杜绝跨集泄漏），按类别分层保证分布一致。
  4. 写出 HuggingFace Dataset（save_to_disk），字段：
       image       224×224 RGB（datasets.Image）
       instruction 自然语言指令字符串
       action      归一化 4 维 float32 [-1,1]

数据落在仓库外，不入 git。

用法（lerobot 环境）:
  python vla_finetune/data/build_hf_dataset.py \
      --base /data/wzh/robot/datasets/vla_pushcube_25/raw_demos_vla.npz \
      --variants /data/wzh/robot/datasets/vla_pushcube_finetune/variants \
      --out /data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from PIL import Image as PILImage

IMAGE_SIZE = 224
ACTION_KEYS = ["dx", "dy", "dz", "gripper"]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute_norm(actions):
    lo = actions.min(axis=0).astype(np.float64)
    hi = actions.max(axis=0).astype(np.float64)
    span = hi - lo
    const_dims = [int(j) for j in range(len(span)) if span[j] < 1e-8]
    return lo, hi, span, const_dims


def normalize(actions, lo, hi, span):
    out = np.zeros_like(actions, dtype=np.float32)
    for j in range(actions.shape[1]):
        if span[j] > 1e-8:
            out[:, j] = (2.0 * (actions[:, j] - lo[j]) / span[j] - 1.0).astype(np.float32)
    return out


def iter_physical_trajectories(base_path, variants_dir, aug_mod, tpl_mod, iv_mod):
    """yield dict(source_kind, source_id, category, images[HWC uint8], actions[4d])"""
    with np.load(base_path, allow_pickle=True) as d:
        images = d["images"]
        actions = d["actions"]
        lengths = d["episode_lengths"]
        tids = d["episode_template_ids"]
    starts = np.concatenate([[0], np.cumsum(lengths)])
    for ep in range(len(lengths)):
        tid = str(tids[ep])
        cat = str(tpl_mod.get_category(tid))
        ins_list = [str(tpl_mod.get_instruction(tid))]
        ins_list += list(aug_mod.PARAPHRASES.get(tid, []))
        yield {
            "source_kind": "base", "source_id": f"base_{ep:03d}",
            "category": cat, "template_id": tid,
            "images": images[starts[ep]:starts[ep + 1]],
            "actions": actions[starts[ep]:starts[ep + 1]],
            "instructions": ins_list,
        }

    for name, cfg in iv_mod.VARIANTS.items():
        path = os.path.join(variants_dir, f"variant_{name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as d:
            imgs = d["images"]
            acts = d["actions"]
            lengths = d["episode_lengths"]
        starts = np.concatenate([[0], np.cumsum(lengths)])
        for ep in range(len(lengths)):
            yield {
                "source_kind": "variant", "source_id": f"{name}_{ep:03d}",
                "category": cfg["category"], "template_id": name.upper(),
                "images": imgs[starts[ep]:starts[ep + 1]],
                "actions": acts[starts[ep]:starts[ep + 1]],
                "instructions": list(cfg["instructions"]),
            }


def grouped_split(trajs, seed=42, ratios=(0.8, 0.1, 0.1)):
    """按物理轨迹分组的分层 8:1:1 划分；同组全进同一 split。"""
    by_cat = defaultdict(list)
    for t in trajs:
        by_cat[t["category"]].append(t)
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for cat in sorted(by_cat):
        group = list(by_cat[cat])
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * ratios[0]))
        n_val = max(1, int(round(n * ratios[1])))
        n_train = min(n_train, n - n_val - 1)  # 至少留 1 条给 test
        split["train"].extend(group[:n_train])
        split["val"].extend(group[n_train:n_train + n_val])
        split["test"].extend(group[n_train + n_val:])
    for s in split:
        rng.shuffle(split[s])
    return split


def build_rows(trajs, lo, hi, span):
    rows = []
    for t in trajs:
        acts = normalize(t["actions"], lo, hi, span)
        n = len(acts)
        for ins in t["instructions"]:
            for i in range(n):
                pil = PILImage.fromarray(t["images"][i].astype(np.uint8), mode="RGB").copy()
                rows.append({
                    "image": pil,
                    "instruction": str(ins),
                    "action": acts[i].astype(np.float32),
                    "_source_id": t["source_id"],
                    "_category": t["category"],
                    "_template_id": t["template_id"],
                })
    return rows


def rows_to_dataset(rows):
    return Dataset.from_list(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        features=Features({
            "image": HFImage(),
            "instruction": Value("string"),
            "action": Sequence(Value("float32"), length=4),
        }),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/data/wzh/robot/datasets/vla_pushcube_25/raw_demos_vla.npz")
    ap.add_argument("--variants", default="/data/wzh/robot/datasets/vla_pushcube_finetune/variants")
    ap.add_argument("--repo", default="/data/wzh/robot/vla-world-model-lab")
    ap.add_argument("--out", default="/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    aug = _load_module("_aug", os.path.join(args.repo, "scripts", "augment_vla_dataset.py"))
    tpl = _load_module("_tpl", os.path.join(args.repo, "vla", "data", "instruction_templates.py"))
    iv = _load_module("_iv", os.path.join(args.repo, "vla_finetune", "data", "instruction_variants.py"))

    trajs = list(iter_physical_trajectories(args.base, args.variants, aug, tpl, iv))
    print(f"[源] 物理轨迹 {len(trajs)} 条（base={sum(1 for t in trajs if t['source_kind']=='base')}, "
          f"variant={sum(1 for t in trajs if t['source_kind']=='variant')}）")

    all_actions = np.concatenate([t["actions"] for t in trajs], axis=0).astype(np.float64)
    lo, hi, span, const_dims = compute_norm(all_actions)
    print(f"[动作] min={np.round(lo,5).tolist()} max={np.round(hi,5).tolist()} 常量维={const_dims}")

    split = grouped_split(trajs, args.seed)
    summary = {
        "format": "huggingface_dataset", "image_size": IMAGE_SIZE,
        "action_dim": 4, "action_keys": ACTION_KEYS,
        "norm": {"method": "min-max -> [-1,1]", "min": lo.tolist(),
                 "max": hi.tolist(), "span": span.tolist(), "constant_dims": const_dims},
        "split_strategy": "physical-trajectory grouped, stratified by category, 8:1:1",
        "seed": args.seed, "splits": {},
    }
    os.makedirs(args.out, exist_ok=True)
    for s in ("train", "val", "test"):
        rows = build_rows(split[s], lo, hi, span)
        ds = rows_to_dataset(rows)
        save_dir = os.path.join(args.out, s)
        ds.save_to_disk(save_dir)
        src_ids = sorted(set(r["_source_id"] for r in rows))
        summary["splits"][s] = {
            "physical_trajectories": len(split[s]),
            "language_conditioned_trajectories": len(set(
                (r["_source_id"], r["instruction"]) for r in rows)),
            "frames": len(ds),
            "categories": dict(Counter(r["_category"] for r in rows)),
            "source_ids": src_ids,
        }
        print(f"[{s}] phys={len(split[s])} lang_traj="
              f"{summary['splits'][s]['language_conditioned_trajectories']} frames={len(ds)}")

    with open(os.path.join(args.out, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"version": "finetune-1.0", "action": summary["norm"]}, f,
                  ensure_ascii=False, indent=2)
    print("[完成] dataset_summary.json / norm_stats.json 已保存")


if __name__ == "__main__":
    main()
