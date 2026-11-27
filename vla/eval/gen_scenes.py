# -*- coding: utf-8 -*-
"""生成三类泛化测试场景（单一真源，VLA 与 ACT 共用同一份场景文件）。

三类：
  * instruction：指令泛化用固定场景（默认起点/目标），只换语言指令；
  * start_pos：目标固定，立方体起点在默认值 ±5 cm 内随机 20 次；
  * target_pos：起点固定，目标在默认值前后左右（x/y 各 ±5 cm）共 20 个场景。
每类场景标注 kind / scene_id / cube_xy / target_xy，保证两算法读取逐字节一致。
"""
import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_CUBE = np.array([0.62, -0.12])
DEFAULT_TARGET = np.array([0.82, -0.12])


def make_scenes(seed=20261127, n_start=20, target_offset=0.05):
    rng = np.random.default_rng(seed)
    scenes = []

    # 指令泛化：固定默认场景，3 个重复场景（配合每条指令 3 回合）
    for i in range(3):
        scenes.append({
            "kind": "instruction", "scene_id": i,
            "cube_xy": DEFAULT_CUBE.tolist(),
            "target_xy": DEFAULT_TARGET.tolist(),
        })

    # 初始位置泛化：目标固定，起点 ±5 cm 随机 20 次
    for i in range(n_start):
        cube = DEFAULT_CUBE + rng.uniform(-0.05, 0.05, size=2)
        scenes.append({
            "kind": "start_pos", "scene_id": i,
            "cube_xy": [round(float(v), 6) for v in cube],
            "target_xy": DEFAULT_TARGET.tolist(),
        })

    # 目标位置泛化：起点固定，目标前后左右偏移共 20 场景
    # 四个方向各 5 个，在 ±5 cm 范围内带轻微抖动
    directions = [
        ("right", np.array([+1.0, 0.0])),
        ("left", np.array([-1.0, 0.0])),
        ("back", np.array([0.0, +1.0])),
        ("front", np.array([0.0, -1.0])),
    ]
    sid = 0
    for name, d in directions:
        for _ in range(5):
            dist = rng.uniform(0.03, target_offset)
            off = d * dist + rng.uniform(-0.008, 0.008, size=2)
            target = DEFAULT_TARGET + off
            scenes.append({
                "kind": "target_pos", "scene_id": sid,
                "direction": name,
                "cube_xy": DEFAULT_CUBE.tolist(),
                "target_xy": [round(float(v), 6) for v in target],
            })
            sid += 1

    payload = {
        "name": "push_cube_vla_generalization_scenes",
        "master_seed": seed,
        "default_cube_xy": DEFAULT_CUBE.tolist(),
        "default_target_xy": DEFAULT_TARGET.tolist(),
        "ranges": {
            "start_offset_m": 0.05, "target_offset_m": target_offset,
            "n_start_scenes": n_start, "n_target_scenes": sid,
            "success_radius_m": 0.03,
        },
        "scenes": scenes,
    }
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    payload = make_scenes(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    by_kind = {}
    for s in payload["scenes"]:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
    print("scenes ->", out, "total", len(payload["scenes"]), by_kind)
    print("SCENES_DONE")


if __name__ == "__main__":
    main()
