"""生成 VLA 数据集 v1.0 统计报告（Markdown + JSON）。

覆盖：轨迹数量、总帧数、指令分布（类别/模板）、动作范围、图像统计、
质量抽查结果、验证集基线指标（帧数/清晰度/动作范围），以及增强配置。
"""

import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np


def sharpness(img_chw):
    arr = np.asarray(img_chw)
    if arr.ndim == 3 and arr.shape[0] == 3:
        hwc = (np.transpose(np.clip(arr, 0, 1), (1, 2, 0)) * 255).astype(np.uint8)
    else:
        hwc = arr.astype(np.uint8)
    g = cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def describe(npz_path):
    with np.load(npz_path, allow_pickle=True) as d:
        acts = d["actions"]
        imgs = d["images"]
        instrs = d["instructions"]
        eps = d["episode_ids"]
        cats = d["categories"]
        tids = d["template_ids"]
    ep_list = sorted(set(int(e) for e in eps))
    sharp = [sharpness(imgs[i]) for i in range(0, len(imgs), max(1, len(imgs)//100))]
    # 类别/模板是轨迹级常量：按 episode 去重取每轨迹首帧精确统计（避免步长采样偏差）
    _, first_idx = np.unique(eps, return_index=True)
    cats_per_ep = cats[first_idx]
    tids_per_ep = tids[first_idx]
    return {
        "episodes": len(ep_list),
        "frames": int(len(imgs)),
        "frames_per_episode": {
            "min": int(min(np.sum(eps == e) for e in ep_list)),
            "max": int(max(np.sum(eps == e) for e in ep_list)),
            "mean": round(float(np.mean([np.sum(eps == e) for e in ep_list])), 2),
        },
        "action_range": {"min": acts.min(axis=0).tolist(), "max": acts.max(axis=0).tolist(),
                         "mean_abs": np.abs(acts).mean(axis=0).tolist()},
        "instruction_unique": int(len(set(str(x) for x in instrs))),
        "category_dist": dict(Counter(str(x) for x in cats_per_ep)),
        "template_dist": dict(Counter(str(x) for x in tids_per_ep)),
        "image_sharpness": {"mean": round(float(np.mean(sharp)), 2),
                            "min": round(float(np.min(sharp)), 2),
                            "max": round(float(np.max(sharp)), 2)},
        "image_pixel_range": [float(imgs.min()), float(imgs.max())],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10", default="/data/wzh/robot/datasets/vla_pushcube_25/v1.0")
    ap.add_argument("--audit", default="/data/wzh/robot/datasets/vla_pushcube_25/quality_audit_v1.json")
    ap.add_argument("--out", default="/data/wzh/robot/vla-world-model-lab/results/vla_dataset/stats_report_v1.md")
    args = ap.parse_args()

    train = describe(os.path.join(args.v10, "train_v1.npz"))
    val = describe(os.path.join(args.v10, "val_v1.npz"))
    aug = json.load(open(os.path.join(args.v10, "augmentation_summary.json"), encoding="utf-8"))
    audit = json.load(open(args.audit, encoding="utf-8"))

    out_json = os.path.join(args.v10, "stats_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "train": train, "val": val,
                   "augmentation": aug, "quality_audit": audit}, f, ensure_ascii=False, indent=2)

    lines = [
        "# VLA 数据集 v1.0 统计报告",
        "",
        "本报告由 `scripts/generate_vla_stats_report.py` 自动生成，数据源为 v1.0 数据集与质量抽查结果。",
        "",
        "## 1. 总体规模",
        "",
        "| 划分 | 轨迹数 | 帧数 | 单轨迹帧数(min/mean/max) | 指令数 |",
        "|---|---|---|---|---|",
        f"| 训练集 v1.0 | {train['episodes']} | {train['frames']} | "
        f"{train['frames_per_episode']['min']}/{train['frames_per_episode']['mean']}/{train['frames_per_episode']['max']} | "
        f"{train['instruction_unique']} |",
        f"| 验证集（基线，不变） | {val['episodes']} | {val['frames']} | "
        f"{val['frames_per_episode']['min']}/{val['frames_per_episode']['mean']}/{val['frames_per_episode']['max']} | "
        f"{val['instruction_unique']} |",
        "",
        f"训练集由 {aug['train']['source_episodes']} 条原始轨迹经增强扩为 "
        f"{aug['train']['output_blocks']} 条轨迹块（原始 + 图像增强 + 每条 2 个同义指令副本）。",
        "",
        "## 2. 指令分布",
        "",
        f"训练集类别分布：{json.dumps(train['category_dist'], ensure_ascii=False)}",
        "",
        f"训练集模板分布：{json.dumps(train['template_dist'], ensure_ascii=False)}",
        "",
        f"验证集类别分布：{json.dumps(val['category_dist'], ensure_ascii=False)}",
        "",
        "指令同义增强：20 条模板各配 2 条语义等价改写（与原指令共 3 种表述），"
        "均指向「把立方体送到红色目标区」这一物理任务。",
        "",
        "## 3. 动作范围（min-max 归一化，[-1,1]）",
        "",
        "| 划分 | 维 | min | max | mean(|x|) |",
        "|---|---|---|---|---|",
    ]
    dims = ["dx", "dy", "dz", "gripper"]
    for tag, st in (("训练", train), ("验证", val)):
        r = st["action_range"]
        for j, name in enumerate(dims):
            lines.append(f"| {tag} | {name} | {r['min'][j]:.4f} | {r['max'][j]:.4f} | {r['mean_abs'][j]:.4f} |")
    lines += [
        "",
        "## 4. 图像统计与基线指标",
        "",
        f"- 训练集像素范围：[{train['image_pixel_range'][0]:.4f}, {train['image_pixel_range'][1]:.4f}]，"
        f"清晰度（Laplacian 方差）均值 {train['image_sharpness']['mean']}，最小 {train['image_sharpness']['min']}。",
        f"- **验证集基线**：{val['frames']} 帧 / {val['episodes']} 回合，"
        f"清晰度均值 {val['image_sharpness']['mean']}、最小 {val['image_sharpness']['min']}，"
        f"像素范围 [{val['image_pixel_range'][0]:.4f}, {val['image_pixel_range'][1]:.4f}]，"
        f"动作范围见上表。验证集未做任何增强，作为微调效果对比基线。",
        "",
        "## 5. 质量抽查",
        "",
        f"- 随机种子 {audit['seed']}，抽查轨迹 "
        f"{[a['episode_id'] for a in audit['audited_episodes']]}，逐帧核对图像/动作/指令。",
    ]
    for a in audit["audited_episodes"]:
        lines.append(
            f"  - ep{a['episode_id']}：{a['frames']} 帧，指令一致={a['instruction_consistent']}，"
            f"帧索引连续={a['frame_index_continuous']}，物理空间相邻动作最大跳变="
            f"{a['action_max_step_jump_m']:.5f}m（量程内），"
            f"模糊帧={a['blurry_frames']} → **{a['verdict']}**"
        )
    g = audit["global_stats"]
    lines += [
        f"- 全局：指令模板错位 {g['instruction_template_mismatches']}，动作越界 "
        f"{g['action_out_of_range_frames']}，非有限值 {g['action_nonfinite_frames']}，"
        f"清晰度均值 {g['image_sharpness_mean']:.2f}。",
        f"- 质量抽查状态：**{audit['status']}**",
        "",
        "## 6. 增强配置",
        "",
        f"- 图像增强：亮度 U{aug['train']['params']['brightness_range']}，"
        f"对比度 U{aug['train']['params']['contrast_range']}，"
        f"高斯噪声 sigma U{aug['train']['params']['noise_sigma_range']}（每条轨迹参数固定，帧内一致）。",
        f"- 指令增强：每条轨迹 2 条同义改写，seed={aug['train']['params']['seed']}。",
        "- 增强仅作用于训练集；验证集保持原始状态作为基线。",
        "",
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("saved", args.out)
    print("saved", out_json)


if __name__ == "__main__":
    main()
