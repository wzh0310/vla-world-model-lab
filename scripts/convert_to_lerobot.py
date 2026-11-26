"""把预处理数据集转换为 LeRobot v0.7.0 标准格式（Parquet）。

在 lerobot 环境运行。字段严格对齐用户要求与官方规范：
    episode_index                 回合编号（save_episode 顺序自动生成）
    frame_index                   帧序号（writer 自动）
    timestamp                     时间戳（writer 按 fps 自动）
    observation.images.main       单帧 RGB 图像
    observation.language_instruction  指令文本（回合级，帧内共享）
    action                        归一化动作向量（min-max -> [-1,1] 的 4 维）

另附官方自动注入的 index/task_index/task 字段。不引入未经验证的字段（如 state），
避免虚构数据。训练集与验证集分别写成两个独立 LeRobotDataset 目录，保证划分清晰、
可独立加载。

用法（lerobot 环境）:
    python scripts/convert_to_lerobot.py \
        --src /data/wzh/robot/datasets/vla_pushcube_25/preprocessed \
        --out /data/wzh/robot/datasets/vla_pushcube_25/lerobot
"""

import argparse
import json
import os

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "observation.images.main": {
        "dtype": "image", "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.language_instruction": {
        "dtype": "string", "shape": (1,), "names": None,
    },
    "action": {"dtype": "float32", "shape": (4,), "names": None},
}


def to_uint8_hwc(img_chw_float):
    """[3,224,224] float [0,1] -> [224,224,3] uint8（image feature 期望）。"""
    arr = np.clip(np.asarray(img_chw_float, dtype=np.float32), 0.0, 1.0)
    arr = np.round(arr * 255.0).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def convert_split(split_name, npz_path, out_root, fps=50):
    with np.load(npz_path, allow_pickle=True) as d:
        images = d["images"]
        actions = d["actions"]
        instrs = d["instructions"]
        ep_ids = d["episode_ids"]
        frame_idx = d["frame_indices"]

    repo_id = f"wzh/pushcube_vla_{split_name}"
    root = os.path.join(out_root, split_name)
    if os.path.exists(root):
        import shutil
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=FEATURES,
        root=root, robot_type="panda", use_videos=False,
    )

    unique_eps = sorted(set(int(e) for e in ep_ids))
    counts = []
    for ep in unique_eps:
        idxs = np.where(ep_ids == ep)[0]
        # 按 frame_index 排序，保证时序对齐
        idxs = sorted(idxs, key=lambda i: int(frame_idx[i]))
        for i in idxs:
            frame = {
                "observation.images.main": to_uint8_hwc(images[i]),
                "observation.language_instruction": str(instrs[i]),
                "action": np.asarray(actions[i], dtype=np.float32),
                "task": str(instrs[i]),
            }
            ds.add_frame(frame)
        ds.save_episode()
        counts.append((ep, len(idxs)))

    ds.finalize()
    print(f"[{split_name}] episodes={len(unique_eps)} frames={len(images)} -> {root}")
    print(f"[{split_name}] per-episode frames: {counts[:5]} ...")
    return {
        "split": split_name,
        "repo_id": repo_id,
        "root": root,
        "episodes": len(unique_eps),
        "frames": int(len(images)),
        "source_episode_ids": unique_eps,
        "fps": fps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/wzh/robot/datasets/vla_pushcube_25/preprocessed")
    ap.add_argument("--out", default="/data/wzh/robot/datasets/vla_pushcube_25/lerobot")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = {"format": "lerobot_v0.7.0", "use_videos": False,
               "action_dim": 4, "image_size": 224, "splits": []}
    for split in ("train", "val"):
        npz = os.path.join(args.src, f"{split}.npz")
        info = convert_split(split, npz, args.out)
        summary["splits"].append(info)

    with open(os.path.join(args.out, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[完成] conversion_summary.json 已保存")


if __name__ == "__main__":
    main()
