#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验三元组中间数据集：指令与轨迹一一对应 + 回合级共享 + 索引一致。

检查项：
  1. 全局字段齐全、图像/动作/状态/时间戳对齐；
  2. 每条轨迹恰好一条指令，指令-轨迹一一对应（无重复绑定、无遗漏）；
  3. 回合级共享：frame_instructions 在同一 episode 内完全相同；
  4. 每条独立 epNNNN.npz 与全局切片一致；
  5. 索引表 JSON/CSV 与 npz 一致，task_index 映射正确；
  6. 指令均在模板库内、类别正确。

用法：
  python scripts/verify_vla_triplets.py --root /data/wzh/robot/datasets/vla_pushcube_25/triplets
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
    spec = importlib.util.spec_from_file_location(
        "_itpl_vt", os.path.join(REPO, "vla", "data", "instruction_templates.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_itpl_vt"] = m
    spec.loader.exec_module(m)
    return m


def check(root):
    itpl = load_templates()
    d = np.load(os.path.join(root, "triplets.npz"), allow_pickle=True)
    needed = ["images", "actions", "states", "timestamps", "episode_lengths",
              "episode_ids", "episode_instructions", "episode_template_ids",
              "episode_categories", "episode_task_indices", "frame_episode_ids",
              "frame_instructions", "frame_task_indices", "fps", "total_frames"]
    missing = [k for k in needed if k not in d.files]
    assert not missing, f"missing fields: {missing}"

    imgs, acts, sts, ts = d["images"], d["actions"], d["states"], d["timestamps"]
    lens, n_ep = d["episode_lengths"], len(d["episode_ids"])
    n_fr = int(lens.sum())
    assert imgs.shape[0] == acts.shape[0] == sts.shape[0] == ts.shape[0] == n_fr
    assert int(d["total_frames"]) == n_fr
    assert list(d["episode_ids"]) == list(range(n_ep))
    print(f"[vtrip] {n_ep} episodes / {n_fr} frames; shapes OK")

    instr = [str(x) for x in d["episode_instructions"]]
    tids = [str(x) for x in d["episode_template_ids"]]
    cats = [str(x) for x in d["episode_categories"]]
    assert len(instr) == len(tids) == len(cats) == n_ep
    # 一一对应：指令在模板库内、类别匹配、模板与指令严格对应
    for ep in range(n_ep):
        assert tids[ep] in itpl.TEMPLATES, f"bad template id ep{ep}"
        assert itpl.get_instruction(tids[ep]) == instr[ep], f"instruction mismatch ep{ep}"
        assert itpl.get_category(tids[ep]) == cats[ep], f"category mismatch ep{ep}"
    print(f"[vtrip] one-to-one OK: {len(set(tids))} unique templates, "
          f"{len(set(cats))} categories")

    # 回合级共享：同一 episode 内帧级指令完全相同
    frame_ep = d["frame_episode_ids"]
    frame_instr = [str(x) for x in d["frame_instructions"]]
    frame_task = d["frame_task_indices"]
    assert len(frame_ep) == n_fr and len(frame_instr) == n_fr
    starts = np.cumsum(np.insert(lens.astype(np.int64), 0, 0))
    task_idx = d["episode_task_indices"]
    for ep in range(n_ep):
        s, e = int(starts[ep]), int(starts[ep + 1])
        seg = frame_instr[s:e]
        assert all(x == instr[ep] for x in seg), f"instruction not shared in ep{ep}"
        assert np.all(frame_ep[s:e] == ep), f"frame_episode mismatch ep{ep}"
        assert np.all(frame_task[s:e] == task_idx[ep]), f"frame_task mismatch ep{ep}"
    print("[vtrip] episode-level sharing OK (all frames within a demo share one instruction)")

    # 每条独立文件与全局切片一致
    ep_dir = os.path.join(root, "episodes")
    for ep in range(n_ep):
        fp = os.path.join(ep_dir, f"ep{ep:04d}.npz")
        assert os.path.exists(fp), f"missing {fp}"
        one = np.load(fp, allow_pickle=True)
        s, e = int(starts[ep]), int(starts[ep + 1])
        assert str(one["instruction"]) == instr[ep], f"ep file instruction mismatch {ep}"
        assert one["images"].shape[0] == lens[ep], f"ep file length mismatch {ep}"
        assert np.array_equal(one["images"], imgs[s:e]), f"ep file image mismatch {ep}"
        assert np.allclose(one["actions"], acts[s:e]), f"ep file action mismatch {ep}"
        assert int(one["episode_id"]) == ep
    print("[vtrip] per-episode files OK (25/25 match global slices)")

    # 索引表 JSON 与 CSV 一致
    with open(os.path.join(root, "triplets_index.json"), encoding="utf-8") as f:
        j = json.load(f)
    idx = j["index"]
    assert j["num_episodes"] == n_ep and j["total_frames"] == n_fr
    assert len(idx) == n_ep
    for ep in range(n_ep):
        assert idx[ep]["episode_id"] == ep
        assert idx[ep]["instruction"] == instr[ep]
        assert idx[ep]["template_id"] == tids[ep]
        assert idx[ep]["category"] == cats[ep]
        assert idx[ep]["task_index"] == int(task_idx[ep])
        assert idx[ep]["traj_length"] == int(lens[ep])
        assert idx[ep]["action_dim"] == acts.shape[1]
    with open(os.path.join(root, "triplets_index.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_ep
    for ep in range(n_ep):
        assert rows[ep]["instruction"] == instr[ep]
        assert int(rows[ep]["traj_length"]) == int(lens[ep])
        assert int(rows[ep]["action_dim"]) == acts.shape[1]
    print("[vtrip] index JSON/CSV OK (episode_id/instruction/length/action_dim)")

    # tasks.jsonl
    with open(os.path.join(root, "tasks.jsonl"), encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    assert len(tasks) == len(itpl.template_ids())
    for t in tasks:
        assert itpl.get_instruction(t["template_id"]) == t["instruction"]
    print(f"[vtrip] tasks.jsonl OK ({len(tasks)} task indices)")

    summary = {
        "status": "PASS", "episodes": n_ep, "total_frames": n_fr,
        "templates": len(set(tids)), "categories": sorted(set(cats)),
        "action_dim": int(acts.shape[1]), "assign_mode": j.get("assign_mode"),
        "one_to_one": True, "episode_level_instruction": True,
    }
    with open(os.path.join(root, "verify_result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[vtrip] ALL CHECKS PASSED ->", json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/data/wzh/robot/datasets/vla_pushcube_25/triplets")
    args = p.parse_args()
    check(args.root)


if __name__ == "__main__":
    main()
