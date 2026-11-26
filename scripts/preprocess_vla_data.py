"""把三元组中间数据集预处理为完全匹配 OpenVLA 输入输出的格式。

处理内容
--------
1. 图像：显式 resize 224×224（bicubic，与 OpenVLA resize-naive 一致）、
   RGB / CHW、像素 /255 归一化到 [0,1]。双视觉主干的 mean/std 写入参数文件，
   Dataset 在读取时据此派生 DINOv2 / SigLIP 两个视图，与推理端严格一致。
2. 动作：4 维增量 [dx,dy,dz,gripper] 做数据集级 min-max 归一化到 [-1,1]；
   常量维（gripper 全 0）单独标记，避免除零。min/max 落盘供微调与推理复用。
3. 划分：按指令类别分层抽取，每类各取 1 条组成 5 条验证集，其余 20 条训练，
   保证训练/验证的类别分布一致、可复现。
4. 产出 train.npz / val.npz / norm_stats.json / splits.json，供 PyTorch
   Dataset + DataLoader 直接加载。

用法
----
    python scripts/preprocess_vla_data.py \
        --src /data/wzh/robot/datasets/vla_pushcube_25/triplets \
        --out /data/wzh/robot/datasets/vla_pushcube_25/preprocessed
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

# OpenVLA 双视觉主干归一化口径，实读自 openvla-7b/preprocessor_config.json
VIEW_PARAMS = {
    "dino": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "siglip": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
}
ACTION_KEYS = ["dx", "dy", "dz", "gripper"]
IMAGE_SIZE = 224


def load_source(src):
    """加载三元组全局 npz 与索引表，返回按 episode_id 排序的记录。"""
    with np.load(os.path.join(src, "triplets.npz"), allow_pickle=True) as d:
        arrays = {k: d[k] for k in d.files}
    with open(os.path.join(src, "triplets_index.json"), encoding="utf-8") as f:
        idx = json.load(f)
    rows = sorted(idx["index"], key=lambda r: r["episode_id"])
    return arrays, rows


def stratified_split(rows, n_val=5, seed=42):
    """按指令类别分层：每类各抽 1 条做验证，其余训练，保证分布一致。"""
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(int(r["episode_id"]))
    cats = sorted(by_cat)
    if len(cats) < n_val:
        raise ValueError(f"类别数 {len(cats)} < 验证条数 {n_val}，无法每类各取 1 条")

    rng = np.random.default_rng(seed)
    # 每个类别内部乱序后取第一条，固定 seed 可复现
    val_ids = []
    for cat in cats:
        cand = np.array(sorted(by_cat[cat]))
        rng.shuffle(cand)
        val_ids.append(int(cand[0]))
    val_ids = sorted(val_ids)[:n_val]
    val_set = set(val_ids)
    train_ids = [int(r["episode_id"]) for r in rows if int(r["episode_id"]) not in val_set]
    return train_ids, val_ids


def resize_rgb(img):
    """HWC uint8 -> 224×224 RGB uint8（显式 resize，锁定口径）。"""
    pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
    if pil.size != (IMAGE_SIZE, IMAGE_SIZE):
        pil = pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


def preprocess_images(images):
    """[N,H,W,3] uint8 -> [N,3,224,224] float32，像素 [0,1]，RGB/CHW。"""
    out = np.empty((len(images), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, im in enumerate(images):
        rgb = resize_rgb(im)
        out[i] = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return out


def compute_action_stats(actions):
    """数据集级 min/max 与常量维标记。"""
    lo = actions.min(axis=0).astype(np.float64)
    hi = actions.max(axis=0).astype(np.float64)
    span = hi - lo
    constant_dims = [int(j) for j in range(len(span)) if span[j] < 1e-8]
    return lo, hi, span, constant_dims


def normalize_actions(actions, lo, hi, span):
    """min-max -> [-1,1]：x_norm = 2*(x-lo)/(hi-lo)-1；常量维置 0。"""
    out = np.zeros_like(actions, dtype=np.float32)
    for j in range(actions.shape[1]):
        if span[j] > 1e-8:
            out[:, j] = (2.0 * (actions[:, j] - lo[j]) / span[j] - 1.0).astype(np.float32)
    return out


def build_split_arrays(arrays, rows, episode_ids):
    """按给定 episode_id 集合展开帧级数组。"""
    keep = set(int(e) for e in episode_ids)
    lengths = arrays["episode_lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)])
    per_ep = arrays["episode_instructions"]
    per_tid = arrays["episode_template_ids"]
    per_cat = arrays["episode_categories"]
    per_task = arrays["episode_task_indices"]

    img_idx, act_idx = [], []
    instrs, ep_ids, frame_idx, tasks, cats, tids = [], [], [], [], [], []
    for eid in sorted(keep):
        s, t = int(starts[eid]), int(starts[eid + 1])
        for k in range(s, t):
            img_idx.append(k)
            act_idx.append(k)
            instrs.append(str(per_ep[eid]))
            ep_ids.append(eid)
            frame_idx.append(k - s)
            tasks.append(int(per_task[eid]))
            cats.append(str(per_cat[eid]))
            tids.append(str(per_tid[eid]))
    return {
        "images": arrays["images"][img_idx],
        "actions": arrays["actions"][act_idx],
        "instructions": np.asarray(instrs, dtype=object),
        "episode_ids": np.asarray(ep_ids, dtype=np.int64),
        "frame_indices": np.asarray(frame_idx, dtype=np.int64),
        "task_indices": np.asarray(tasks, dtype=np.int64),
        "categories": np.asarray(cats, dtype=object),
        "template_ids": np.asarray(tids, dtype=object),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/wzh/robot/datasets/vla_pushcube_25/triplets")
    ap.add_argument("--out", default="/data/wzh/robot/datasets/vla_pushcube_25/preprocessed")
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    arrays, rows = load_source(args.src)
    print(f"[源] {len(rows)} 条轨迹 / {len(arrays['images'])} 帧")

    # 图像预处理（全数据集统一口径）
    images = preprocess_images(arrays["images"])
    print(f"[图像] {images.shape}, dtype={images.dtype}, "
          f"范围=[{images.min():.4f}, {images.max():.4f}]")

    # 动作归一化：统计基于全数据集（训练/验证共用同一参数，避免数据泄漏）
    raw_actions = arrays["actions"].astype(np.float64)
    lo, hi, span, const_dims = compute_action_stats(raw_actions)
    actions = normalize_actions(raw_actions, lo, hi, span)
    print(f"[动作] {actions.shape}, 范围=[{actions.min():.4f}, {actions.max():.4f}], "
          f"常量维={const_dims}")

    # 分层划分
    train_ids, val_ids = stratified_split(rows, args.n_val, args.seed)
    print(f"[划分] 训练 {len(train_ids)} 条 / 验证 {len(val_ids)} 条")
    print(f"  验证 episode_ids = {val_ids}")

    # 写出 train/val npz（图像 float32 [0,1]；动作 [-1,1]）
    for name, ids in (("train", train_ids), ("val", val_ids)):
        pack = build_split_arrays(arrays, rows, ids)
        # 用预处理后的图像按帧区间重取（与动作索引顺序一致）
        img_idx = []
        lengths = arrays["episode_lengths"]
        starts = np.concatenate([[0], np.cumsum(lengths)])
        for eid in sorted(ids):
            img_idx.extend(range(int(starts[eid]), int(starts[eid + 1])))
        pack["images"] = images[img_idx]
        pack["actions"] = normalize_actions(
            raw_actions[img_idx], lo, hi, span
        )
        np.savez_compressed(os.path.join(args.out, f"{name}.npz"), **pack)
        print(f"[写出] {name}.npz: {len(pack['images'])} 帧")

    # 归一化参数文件（微调与推理必须复用同一份）
    norm_stats = {
        "version": "1.0",
        "action": {
            "keys": ACTION_KEYS,
            "method": "min-max -> [-1,1]: x_norm = 2*(x-min)/(max-min)-1",
            "min": lo.tolist(),
            "max": hi.tolist(),
            "span": span.tolist(),
            "constant_dims": const_dims,
            "computed_over": "all 25 episodes (928 frames), shared by train/val",
        },
        "image": {
            "size": IMAGE_SIZE,
            "channels": "RGB",
            "layout": "CHW",
            "pixel_range": [0.0, 1.0],
            "resize": "bicubic to 224x224, then /255",
            "views": VIEW_PARAMS,
            "source": "openvla-7b/preprocessor_config.json (means/stds)",
        },
    }
    with open(os.path.join(args.out, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)

    # 划分记录
    def dist(ids):
        m = {r["episode_id"]: r["category"] for r in rows}
        return dict(Counter(m[e] for e in ids))

    splits = {
        "strategy": "per-category stratified (1 episode per category -> validation)",
        "seed": args.seed,
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        "train_category_dist": dist(train_ids),
        "val_category_dist": dist(val_ids),
        "overall_category_dist": dist([int(r["episode_id"]) for r in rows]),
    }
    with open(os.path.join(args.out, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    print("[完成] norm_stats.json / splits.json 已保存")
    print(f"[类别分布] 训练={splits['train_category_dist']}")
    print(f"[类别分布] 验证={splits['val_category_dist']}")


if __name__ == "__main__":
    main()
