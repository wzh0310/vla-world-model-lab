#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可视化抽检：每条轨迹取首帧，按指令类别排布成网格，核对图文对齐。

输出到仓库 results/vla_dataset/ 下（图片入库，npz 数据不入库）。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT_COLOR = {
    "direct": "#2c7fb8", "brief": "#7a5195", "descriptive": "#ef5675",
    "variant": "#ffa600", "reverse": "#009e73",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/data/wzh/robot/datasets/vla_pushcube_25")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = args.out_dir or os.path.join(repo, "results", "vla_dataset")
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(os.path.join(args.root, "raw_demos_vla.npz"), allow_pickle=True)
    imgs, lens = data["images"], data["episode_lengths"]
    tids = [str(x) for x in data["episode_template_ids"]]
    instr = [str(x) for x in data["episode_instructions"]]
    with open(os.path.join(args.root, "instruction_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    cats = [a["category"] for a in manifest["assignments"]]

    # 每类取 4 条（共 20），放不满则补其他类
    picks = []
    for cat in CAT_COLOR:
        idxs = [i for i, c in enumerate(cats) if c == cat][:4]
        picks.extend(idxs)
    picks = picks[:20]

    starts = np.cumsum(np.insert(lens.astype(int), 0, 0))
    fig, axes = plt.subplots(4, 5, figsize=(17, 11))
    for ax, ep in zip(axes.flat, picks):
        frame = imgs[starts[ep]]
        ax.imshow(frame)
        ax.set_title(f"ep{ep} [{tids[ep]} / {cats[ep]}]\n{instr[ep]}",
                     fontsize=8, color=CAT_COLOR.get(cats[ep], "#333"))
        ax.axis("off")
    fig.suptitle("PushCube VLA dataset: 25 demos / 20 templates / 5 categories (first-frame check)",
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "vla_dataset_grid.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[viz] saved {path} ({os.path.getsize(path)} bytes, {len(picks)} frames)")


if __name__ == "__main__":
    main()
