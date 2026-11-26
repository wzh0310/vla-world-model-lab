#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""语言指令标注 + 三元组配对。

把原始轨迹数据（图像 / 动作 / 状态 / 时间戳）与语言指令配对，形成完整的
VLA 数据三元组「instruction - image sequence - action sequence」，并构建
数据索引表与中间格式，供后续 LeRobot 格式化与 LoRA 微调消费。

回合级指令机制（核心）：VLA 采用**单条指令对应整段轨迹**，轨迹内所有帧
共享同一指令，而非逐帧更换。本脚本显式构造 frame->episode 映射并把指令
展开到帧级（frame_instruction 字段），既保留回合级语义又便于逐帧采样。

分配模式：
  * reuse（默认）：沿用采集阶段用模板库 sequence_for_count 做的确定性均衡
    随机分配（25 条覆盖全部 20 条模板），保持与已有数据一致；
  * reshuffle：按给定 seed 重新随机分配，用于演示"随机分配 1 条模板"机制。

输入：raw_demos_vla.npz（仓库外）
输出（仓库外，不入库）：<root>/triplets/
  ├── triplets.npz          全局打包（含帧级指令展开与 episode 映射）
  ├── triplets_index.json   数据索引表（JSON）
  ├── triplets_index.csv    数据索引表（CSV，便于表格查看）
  ├── tasks.jsonl           task_index -> 指令文本映射
  └── episodes/ep0000.npz   每条轨迹独立三元组（25 个，指令与轨迹一一对应）

用法（任意环境，只读 numpy）：
  python scripts/build_vla_triplets.py \
      --raw /data/wzh/robot/datasets/vla_pushcube_25/raw_demos_vla.npz \
      --out /data/wzh/robot/datasets/vla_pushcube_25/triplets
  # 演示重新随机分配：
  python scripts/build_vla_triplets.py --assign-mode reshuffle --seed 7
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_templates():
    path = os.path.join(REPO, "vla", "data", "instruction_templates.py")
    spec = importlib.util.spec_from_file_location("_itpl_tri", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_itpl_tri"] = m
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="/data/wzh/robot/datasets/vla_pushcube_25/raw_demos_vla.npz")
    p.add_argument("--out", default="/data/wzh/robot/datasets/vla_pushcube_25/triplets")
    p.add_argument("--assign-mode", choices=["reuse", "reshuffle"], default="reuse")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    itpl = load_templates()
    d = np.load(args.raw, allow_pickle=True)
    images, actions, states = d["images"], d["actions"], d["states"]
    timestamps, lens = d["timestamps"], d["episode_lengths"]
    fps = int(d["fps"])
    n_ep = len(lens)
    starts = np.cumsum(np.insert(lens.astype(np.int64), 0, 0))

    # ---- 指令分配（回合级：每条轨迹恰好一条）----
    if args.assign_mode == "reuse":
        tids = [str(x) for x in d["episode_template_ids"]]
        instr = [str(x) for x in d["episode_instructions"]]
        assert all(itpl.get_instruction(t) == s for t, s in zip(tids, instr))
        print(f"[triples] assign=reuse, kept pre-assigned templates ({len(set(tids))} unique)")
    else:
        tids = itpl.sequence_for_count(n_ep, seed=args.seed)
        instr = [itpl.get_instruction(t) for t in tids]
        print(f"[triples] assign=reshuffle(seed={args.seed}), "
              f"{len(set(itpl.get_category(t) for t in tids))} categories")

    cats = [itpl.get_category(t) for t in tids]
    # task_index：按模板 id 稳定编号（同一指令共用一个 task_index）
    uniq_tids = list(itpl.template_ids())
    tid2task = {t: i for i, t in enumerate(uniq_tids)}
    task_indices = np.array([tid2task[t] for t in tids], dtype=np.int64)

    # ---- 帧 -> episode 映射 + 帧级指令展开（回合级共享）----
    frame_episode = np.repeat(np.arange(n_ep, dtype=np.int64), lens.astype(np.int64))
    frame_instruction = np.array([instr[e] for e in frame_episode], dtype=object)
    frame_task_index = task_indices[frame_episode]

    os.makedirs(args.out, exist_ok=True)
    ep_dir = os.path.join(args.out, "episodes")
    os.makedirs(ep_dir, exist_ok=True)

    # ---- 每条轨迹独立三元组（一对一）----
    for ep in range(n_ep):
        s, e = int(starts[ep]), int(starts[ep + 1])
        np.savez_compressed(
            os.path.join(ep_dir, f"ep{ep:04d}.npz"),
            instruction=np.array(instr[ep]),
            template_id=np.array(tids[ep]),
            category=np.array(cats[ep]),
            task_index=np.array(task_indices[ep]),
            images=images[s:e], actions=actions[s:e], states=states[s:e],
            timestamps=timestamps[s:e], episode_id=np.int64(ep),
        )

    # ---- 全局打包 ----
    npz_out = os.path.join(args.out, "triplets.npz")
    np.savez_compressed(
        npz_out,
        images=images, actions=actions, states=states, timestamps=timestamps,
        episode_lengths=lens, episode_ids=np.arange(n_ep, dtype=np.int64),
        episode_instructions=np.array(instr, dtype=object),
        episode_template_ids=np.array(tids, dtype=object),
        episode_categories=np.array(cats, dtype=object),
        episode_task_indices=task_indices,
        frame_episode_ids=frame_episode,
        frame_instructions=frame_instruction,
        frame_task_indices=frame_task_index,
        fps=np.int64(fps), total_frames=np.int64(int(lens.sum())),
    )

    # ---- 数据索引表 ----
    index = []
    for ep in range(n_ep):
        s, e = int(starts[ep]), int(starts[ep + 1])
        index.append({
            "episode_id": ep,
            "instruction": instr[ep],
            "template_id": tids[ep],
            "category": cats[ep],
            "task_index": int(task_indices[ep]),
            "traj_length": int(lens[ep]),
            "action_dim": int(actions.shape[1]),
            "state_dim": int(states.shape[1]),
            "image_shape": list(images.shape[1:]),
            "fps": fps,
            "frame_start": s,
            "frame_end": e,
            "final_dist": round(float(d["episode_dist"][ep]), 5),
            "file": f"episodes/ep{ep:04d}.npz",
        })
    with open(os.path.join(args.out, "triplets_index.json"), "w", encoding="utf-8") as f:
        json.dump({"assign_mode": args.assign_mode, "seed": args.seed,
                   "num_episodes": n_ep, "total_frames": int(lens.sum()),
                   "index": index}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "triplets_index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id", "instruction", "template_id",
            "category", "task_index", "traj_length", "action_dim", "state_dim",
            "fps", "frame_start", "frame_end", "final_dist", "file"])
        w.writeheader()
        for row in index:
            w.writerow({k: row[k] for k in w.fieldnames})

    # ---- tasks.jsonl（对接 LeRobot task_index）----
    with open(os.path.join(args.out, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for tid in uniq_tids:
            f.write(json.dumps({"task_index": tid2task[tid],
                                "template_id": tid,
                                "instruction": itpl.get_instruction(tid)},
                               ensure_ascii=False) + "\n")

    # ---- README ----
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# VLA 三元组中间数据集\n\n"
            f"轨迹 {n_ep} 条 / 帧 {int(lens.sum())} 帧 / fps {fps}；"
            f"分配模式 `{args.assign_mode}`。\n\n"
            "## 回合级指令\n"
            "单条指令对应整段轨迹，轨迹内所有帧共享同一指令（非逐帧更换）。"
            "`triplets.npz` 中 `episode_instructions` 为回合级，`frame_instructions` "
            "为按 `frame_episode_ids` 展开的帧级视图，两者语义一致。\n\n"
            "## 文件\n"
            "- `triplets.npz`：全局打包（图像/动作/状态/时间戳 + 回合级与帧级指令）；\n"
            "- `triplets_index.json` / `.csv`：索引表（episode_id、指令文本、轨迹长度、动作维度等）；\n"
            "- `tasks.jsonl`：task_index → 指令文本映射；\n"
            "- `episodes/epNNNN.npz`：每条轨迹独立三元组（指令与轨迹一一对应）。\n\n"
            "中间格式不入库（仓库外），供 LeRobot 格式化与 LoRA 微调消费。\n"
        )

    print(f"[triples] saved {n_ep} episodes / {int(lens.sum())} frames -> {args.out}")
    print(f"[triples] categories={sorted(set(cats))}, templates={len(set(tids))}, "
          f"task_indices={len(uniq_tids)}")


if __name__ == "__main__":
    main()
