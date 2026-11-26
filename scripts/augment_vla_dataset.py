"""数据集增强模块（仅训练集）：图像增强 + 指令同义增强，产出 v1.0 数据集。

图像增强（每条轨迹生成一个增强副本）：
  - 随机亮度：乘性因子 U(0.85, 1.15)
  - 随机对比度：围绕均值缩放 U(0.85, 1.15)
  - 轻微高斯噪声：sigma U(0.0, 0.02)（像素 [0,1] 尺度）
  参数每条轨迹固定（帧内一致），避免同一轨迹内画面闪烁。

指令同义增强：
  - 每个 template_id 预置 2 条语义等价改写（与原指令共 3 种表述），
    均指向「把立方体送到红色目标区」这一物理任务，不引入场景不存在的物体。
  - 对每条轨迹额外复制 2 份，分别用 2 条同义指令替换，图像/动作不变。

产出规则：
  - 验证集完全不变（作为基线），仍用 preprocessed/val.npz；
  - 训练集 v1.0 = 原始 20 条 + 图像增强 20 条 + 指令同义 40 条（每轨迹 2 条改写）
              = 80 条轨迹；帧数按原轨迹展开；
  - 每份样本带 aug_type / source_episode_id / aug_seed 便于追踪。
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageEnhance

# 每个 template_id 的 2 条语义等价改写（英文祈使句，与原指令同一物理任务）
PARAPHRASES = {
    "D01": ["push the cube into the red marked area", "move the block to the red target region"],
    "D02": ["place the cube on the red target", "push the block onto the red circle"],
    "D03": ["move the cube onto the red target disc", "push the blue cube to the red circular mark"],
    "D04": ["push the blue block to the red circle", "move the cube onto the red target spot"],
    "B01": ["push block to red goal", "move block onto the target"],
    "B02": ["push cube to goal", "move cube to red target"],
    "B03": ["move block to red target", "push block onto red disc"],
    "B04": ["push cube to red goal", "move cube onto red target"],
    "S01": ["slide the blue cube on the desk to the circular mark",
            "move the small blue block across the table to the red circle"],
    "S02": ["slide the blue block across the desk to the red circular target",
            "move the blue cube over the tabletop to the red round mark"],
    "S03": ["push the small cube on the table into the red marked circle",
            "slide the small block on the tabletop onto the red target area"],
    "S04": ["move the blue block on the desk to the red circular marker",
            "push the blue cube across the desk onto the red round target"],
    "V01": ["slide the cube forward into the red region", "shift the block ahead to the red target"],
    "V02": ["push the block forward into the red target zone", "slide the cube ahead to the red goal"],
    "V03": ["move the cube toward the red goal area", "shift the block into the red target region"],
    "V04": ["drive the block to the red circular target", "push the cube onto the red circular spot"],
    "R01": ["move the cube away from its start toward the red target",
            "push the block from its initial position to the red goal"],
    "R02": ["slide the cube off its starting spot into the red target area",
            "move the block away from its initial position to the red goal"],
    "R03": ["push the cube from where it begins to the red circular mark",
            "move the block from its starting spot to the red round target"],
    "R04": ["slide the block out of its initial spot onto the red goal",
            "push the cube away from the start into the red target area"],
}


def augment_image(img_chw_float, brightness, contrast, noise_sigma, rng):
    """对单帧 [3,224,224] float[0,1] 做亮度/对比度/高斯噪声增强。"""
    arr = np.clip(img_chw_float, 0, 1)
    # PIL 通道 HWC
    hwc = (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8)
    pil = Image.fromarray(hwc, mode="RGB")
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    out = np.asarray(pil, dtype=np.float32) / 255.0
    if noise_sigma > 0:
        out = out + rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    return np.transpose(out, (2, 0, 1)).astype(np.float32)


def build_v1_split(npz_path, out_path, seed):
    """对训练集 npz 做增强，返回 v1.0 训练集 npz。"""
    with np.load(npz_path, allow_pickle=True) as d:
        images = d["images"]
        actions = d["actions"]
        instrs = d["instructions"]
        ep_ids = d["episode_ids"]
        frame_idx = d["frame_indices"]
        task_idx = d["task_indices"]
        cats = d["categories"]
        template_ids = d["template_ids"]

    rng = np.random.default_rng(seed)
    unique_eps = sorted(set(int(e) for e in ep_ids))
    blocks = []

    # 1) 原始轨迹块
    def make_block(ep, img, ins, aug_type, source_tid):
        idxs = np.where(ep_ids == ep)[0]
        n = len(idxs)
        return {
            "images": img,
            "actions": actions[idxs],
            "instructions": np.array([ins] * n, dtype=object),
            "episode_ids": np.full(n, ep, dtype=np.int64),
            "frame_indices": frame_idx[idxs].copy(),
            "task_indices": task_idx[idxs].copy(),
            "categories": np.full(n, str(cats[idxs[0]]), dtype=object),
            "template_ids": np.full(n, str(source_tid), dtype=object),
            "aug_types": np.full(n, str(aug_type), dtype=object),
            "source_episode_ids": np.full(n, int(ep), dtype=np.int64),
        }

    for ep in unique_eps:
        idxs = np.where(ep_ids == ep)[0]
        tid = str(template_ids[idxs[0]])
        ins = str(instrs[idxs[0]])
        blocks.append(make_block(ep, images[idxs], ins, "original", tid))

        # 2) 图像增强副本（每条轨迹固定增强参数，帧内一致）
        brightness = float(rng.uniform(0.85, 1.15))
        contrast = float(rng.uniform(0.85, 1.15))
        noise_sigma = float(rng.uniform(0.0, 0.02))
        img_aug = np.stack([
            augment_image(images[i], brightness, contrast, noise_sigma, rng) for i in idxs
        ])
        blocks.append(make_block(ep, img_aug, ins, "image_aug", tid))

        # 3) 指令同义副本（图像/动作不变，指令替换）
        for k, para in enumerate(PARAPHRASES.get(tid, [])):
            blocks.append(make_block(ep, images[idxs], para, f"instr_aug_{k+1}", tid))

    # 拼接所有块，重排全局帧 index 由 LeRobot 生成；这里保持 episode_ids 为源编号
    keys = ["images", "actions", "instructions", "episode_ids", "frame_indices",
            "task_indices", "categories", "template_ids", "aug_types", "source_episode_ids"]
    merged = {k: np.concatenate([b[k] for b in blocks], axis=0) for k in keys}
    np.savez_compressed(out_path, **merged)

    return {
        "source_episodes": len(unique_eps),
        "output_blocks": len(blocks),
        "output_frames": int(len(merged["images"])),
        "aug_types": sorted(set(str(x) for x in merged["aug_types"])),
        "params": {"brightness_range": [0.85, 1.15], "contrast_range": [0.85, 1.15],
                   "noise_sigma_range": [0.0, 0.02], "seed": seed},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/wzh/robot/datasets/vla_pushcube_25/preprocessed")
    ap.add_argument("--out", default="/data/wzh/robot/datasets/vla_pushcube_25/v1.0")
    ap.add_argument("--seed", type=int, default=20261126)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    info = build_v1_split(
        os.path.join(args.src, "train.npz"),
        os.path.join(args.out, "train_v1.npz"),
        args.seed,
    )
    # 验证集不变：直接复制引用（不改写内容）
    import shutil
    shutil.copy(os.path.join(args.src, "val.npz"), os.path.join(args.out, "val_v1.npz"))
    shutil.copy(os.path.join(args.src, "norm_stats.json"), os.path.join(args.out, "norm_stats.json"))

    summary = {
        "version": "1.0",
        "train": info,
        "val": {"episodes": 5, "frames": 183, "note": "unchanged baseline"},
        "paraphrases_per_template": 2,
        "templates_covered": len(PARAPHRASES),
    }
    with open(os.path.join(args.out, "augmentation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[增强] 源 {info['source_episodes']} 条 -> {info['output_blocks']} 块 / {info['output_frames']} 帧")
    print(f"[增强类型] {info['aug_types']}")
    print(f"[验证集] 不变（5 条 / 183 帧基线）")
    print(f"[完成] {args.out}")


if __name__ == "__main__":
    main()
