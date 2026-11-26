#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务变体指令表：左右目标 + 每条轨迹 3-5 种同义表述。

承接既有 25 条基础轨迹的 5 类 20 模板，本阶段新增两类空间变体：
  * left  —— 把立方体推到目标区偏左位置（target_xy y 增大方向）
  * right —— 把立方体推到目标区偏右位置（target_xy y 减小方向）

每个变体给 4 条语义等价的英文指令（同一物理目标，仅措辞/方位不同），
用于后续跨指令泛化验证。指令措辞只引用场景中已有物体（蓝色立方体、
红色圆形目标区、桌面），不引入不存在的实体。
"""
from __future__ import annotations

from collections import OrderedDict

# 变体 -> (目标 XY, 类别说明, 4 条同义指令)
# 目标 XY 相对基础 TARGET_XY=(0.82,-0.12) 做 y 方向 +/- 偏移，仍在桌面可达范围内。
VARIANTS = OrderedDict([
    ("left", {
        "target_xy": (0.82, -0.02),
        "description": "推到红色目标区偏左侧",
        "category": "variant_left",
        "instructions": [
            "push the cube to the left side of the red target",
            "move the cube to the left part of the red circle",
            "slide the blue cube onto the left half of the red target area",
            "push the block leftward into the red target zone",
        ],
    }),
    ("right", {
        "target_xy": (0.82, -0.22),
        "description": "推到红色目标区偏右侧",
        "category": "variant_right",
        "instructions": [
            "push the cube to the right side of the red target",
            "move the cube to the right part of the red circle",
            "slide the blue cube onto the right half of the red target area",
            "push the block rightward into the red target zone",
        ],
    }),
])

# 基础轨迹每条至少 3 种表述：原模板 + 2 条同义改写（复用既有 PARAPHRASES 口径）。
# 这里给出通用的额外改写后缀模板，供 build_hf_dataset 对每个 template 生成表述。
GENERIC_PARAPHRASE_HINT = "semantically equivalent rewrites of the same push-to-red-target task"


def variant_names():
    return list(VARIANTS.keys())


def get_variant(name):
    return VARIANTS[name]


def all_instructions():
    """返回 [(variant, category, instruction), ...]，便于统计与校验。"""
    out = []
    for name, cfg in VARIANTS.items():
        for ins in cfg["instructions"]:
            out.append((name, cfg["category"], ins))
    return out


if __name__ == "__main__":
    for name, cfg in VARIANTS.items():
        print(f"[{name}] target_xy={cfg['target_xy']} cat={cfg['category']}")
        for i, ins in enumerate(cfg["instructions"]):
            print(f"  {i}: {ins}")
    print(f"total variants={len(VARIANTS)}, instructions={len(all_instructions())}")
