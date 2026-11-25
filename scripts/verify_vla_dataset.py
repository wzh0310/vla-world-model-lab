#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 VLA 原始数据集（只读 numpy，任意环境可跑）。

检查项：
  1. 必须字段齐全且 shape/dtype 正确；
  2. 轨迹数、帧数范围、总帧数符合方案；
  3. 时间戳回合内从 0 起、单调递增、步长=1/fps；
  4. 指令分配覆盖 5 类、每条模板至少 1 次、图文字段对齐；
  5. 动作 4 维、位移在 ±0.02、夹爪在 [0,1]；
  6. 每条轨迹到达目标（dist<0.03）、起点偏移覆盖 ±0.05。

用法：
  python scripts/verify_vla_dataset.py --root /data/wzh/robot/datasets/vla_pushcube_25
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_templates():
    spec = importlib.util.spec_from_file_location(
        "_itpl", os.path.join(REPO, "vla", "data", "instruction_templates.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_itpl"] = m
    spec.loader.exec_module(m)
    return m


def check(root, expect_episodes=25, min_frames=30, max_frames=40):
    npz_path = os.path.join(root, "raw_demos_vla.npz")
    data = np.load(npz_path, allow_pickle=True)
    needed = ["images", "actions", "states", "timestamps", "episode_lengths",
              "cube_starts", "arm_starts", "episode_instructions",
              "episode_template_ids", "episode_success", "episode_dist", "fps", "task"]
    missing = [k for k in needed if k not in data.files]
    assert not missing, f"missing fields: {missing}"

    imgs, acts, sts = data["images"], data["actions"], data["states"]
    ts, lens = data["timestamps"], data["episode_lengths"]
    fps = int(data["fps"])
    n_ep, n_fr = len(lens), int(lens.sum())
    print(f"[verify] episodes={n_ep} frames={n_fr} fps={fps} task={str(data['task'])}")

    assert n_ep == expect_episodes, f"episodes {n_ep} != {expect_episodes}"
    assert imgs.shape == (n_fr, 224, 224, 3) and imgs.dtype == np.uint8
    assert acts.shape == (n_fr, 4) and acts.dtype == np.float32
    assert sts.shape == (n_fr, 13)
    assert ts.shape == (n_fr,) and lens.dtype == np.int64
    assert np.all((lens >= min_frames) & (lens <= max_frames)), \
        f"frame range violated: {lens.min()}-{lens.max()}"
    assert 600 <= n_fr <= 1100, f"total frames {n_fr} out of expected ~800"

    # 时间戳：回合内从 0 起、等间隔 1/fps、单调递增
    off = 0
    for ep, L in enumerate(lens):
        seg = ts[off:off + int(L)]
        expect = np.arange(int(L), dtype=np.float32) / fps
        assert np.allclose(seg, expect, atol=1e-5), f"timestamp mismatch ep{ep}"
        off += int(L)
    print(f"[verify] timestamps OK (per-episode 0..(L-1)/{fps})")

    # 动作范围：位移 ±0.02、夹爪 [0,1]
    assert np.all(np.abs(acts[:, :3]) <= 0.0201), "xyz action out of ±0.02"
    assert np.all((acts[:, 3] >= -0.001) & (acts[:, 3] <= 1.001)), "gripper out of [0,1]"
    print(f"[verify] action range OK xyz max|.|={np.abs(acts[:,:3]).max():.4f} "
          f"gripper [{acts[:,3].min():.3f},{acts[:,3].max():.3f}]")

    # 指令与模板
    itpl = load_templates()
    tids = [str(x) for x in data["episode_template_ids"]]
    instr = [str(x) for x in data["episode_instructions"]]
    assert len(tids) == n_ep and len(instr) == n_ep
    assert all(itpl.get_category(t) in itpl.CATEGORY_DESC for t in tids), "bad template id"
    assert all(itpl.get_instruction(t) == s for t, s in zip(tids, instr)), "instruction mismatch"
    cats = {itpl.get_category(t) for t in tids}
    assert cats == set(itpl.CATEGORY_DESC.keys()), f"categories not full: {cats}"
    used = set(tids)
    all_t = set(itpl.template_ids())
    print(f"[verify] instructions OK: {len(cats)} categories, {len(used)}/{len(all_t)} templates used")

    # 到达目标 + 起点偏移覆盖
    dist = data["episode_dist"]
    assert np.all(dist < 0.03), f"undone episode dist max={dist.max():.4f}"
    starts = data["cube_starts"]
    xr, yr = starts[:, 0].min(), starts[:, 1].min()
    xM, yM = starts[:, 0].max(), starts[:, 1].max()
    print(f"[verify] success OK: dist max={dist.max():.4f}; "
          f"cube start x[{xr:.3f},{xM:.3f}] y[{yr:.3f},{yM:.3f}]")

    # 与 json 清单交叉核对
    with open(os.path.join(root, "instruction_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["count"] == n_ep and manifest["total_frames"] == n_fr
    assert [a["template_id"] for a in manifest["assignments"]] == tids
    print("[verify] manifest cross-check OK")

    summary = {
        "episodes": n_ep, "total_frames": n_fr, "fps": fps,
        "frames_min": int(lens.min()), "frames_max": int(lens.max()),
        "categories": sorted(cats), "templates_used": len(used),
        "dist_max": round(float(dist.max()), 5),
        "cube_start_x": [round(float(xr), 4), round(float(xM), 4)],
        "cube_start_y": [round(float(yr), 4), round(float(yM), 4)],
        "status": "PASS",
    }
    with open(os.path.join(root, "verify_result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[verify] ALL CHECKS PASSED ->", json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/data/wzh/robot/datasets/vla_pushcube_25")
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--min-frames", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=40)
    args = p.parse_args()
    check(args.root, args.episodes, args.min_frames, args.max_frames)


if __name__ == "__main__":
    main()
