# -*- coding: utf-8 -*-
"""生成左/右跨任务评估场景（VLA 基线与 ACT 共用逐字节一致的几何）。"""
import argparse
import json
from pathlib import Path
import numpy as np

CUBE = [0.62, -0.12]
# 与示教目标同分布但位置不同的横向目标（评估泛化，非记忆）
TEMPLATES = {
    "left": [[0.79, -0.25], [0.81, -0.27], [0.78, -0.23], [0.80, -0.26], [0.82, -0.28]],
    "right": [[0.79, 0.01], [0.81, 0.03], [0.78, 0.00], [0.80, 0.02], [0.82, 0.04]],
}


def make_scenes(seed=20261127, jitter=0.006):
    rng = np.random.default_rng(seed)
    scenes = []
    sid = 0
    for task, tmpl in TEMPLATES.items():
        for base in tmpl:
            j = rng.uniform(-jitter, jitter, size=2)
            target = (np.array(base) + j).round(6).tolist()
            scenes.append({"task": task, "scene_id": sid,
                           "cube_xy": list(CUBE), "target_xy": target})
            sid += 1
    return {"name": "push_cube_left_right_cross_task", "master_seed": seed,
            "cube_start": CUBE, "jitter_m": jitter,
            "instruction_map": {"left": "把方块推到左边", "right": "把方块推到右边"},
            "n_scenes": len(scenes), "scenes": scenes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    payload = make_scenes(args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("scenes ->", out, "n=", payload["n_scenes"])
    print("CROSS_SCENES_DONE")


if __name__ == "__main__":
    main()
