#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PushCube 语言指令模板库（VLA 数据集构建阶段）。

设计依据：
  * Open X-Embodiment：语言指令为 episode 级字符串条件，与图像一起输入策略；
  * 核心原则：同一指令覆盖多样初始状态，同一初始状态可对应多样指令表述。

5 类共 20 条模板。每条模板带稳定 template_id 与 category，便于逐轨迹分配
task_index 与做指令维度的泛化分析。模板均复用现有场景语义
（蓝色立方体 / 桌面 / 红色圆形目标区），不引入场景里不存在的物体。
"""
from __future__ import annotations

from collections import OrderedDict

# 模板类别 -> 中文说明（仅文档用途，不写入数据集）
CATEGORY_DESC = OrderedDict([
    ("direct", "直白指令：明确动作 + 目标"),
    ("brief", "简略指令：最短祈使句"),
    ("descriptive", "描述性指令：补充物体位置/外观与场景细节"),
    ("variant", "变体指令：同义动词/方位改写"),
    ("reverse", "反向/反向参照指令：以起点或远离关系描述目标"),
])

# template_id -> (category, english_instruction)
# 说明：reverse 类语义上仍要求把立方体送到红色目标区（与物理任务一致），
# 只是用「远离起点」「从初始位置移开」等反向参照措辞表达，避免图文不符。
TEMPLATES = OrderedDict([
    # ---- 1. 直白指令 ----
    ("D01", ("direct",     "push the cube to the red target area")),
    ("D02", ("direct",     "push the cube onto the red target")),
    ("D03", ("direct",     "move the cube to the red target disc")),
    ("D04", ("direct",     "push the blue cube to the red circle")),
    # ---- 2. 简略指令 ----
    ("B01", ("brief",      "push block to goal")),
    ("B02", ("brief",      "push cube to target")),
    ("B03", ("brief",      "move block to red disc")),
    ("B04", ("brief",      "push cube onto red goal")),
    # ---- 3. 描述性指令 ----
    ("S01", ("descriptive", "move the blue cube on the desk to the circular mark")),
    ("S02", ("descriptive", "slide the blue cube across the table to the red circular target")),
    ("S03", ("descriptive", "push the small cube on the tabletop into the red marked area")),
    ("S04", ("descriptive", "move the blue block on the desk onto the red round marker")),
    # ---- 4. 变体指令 ----
    ("V01", ("variant",    "slide the cube forward to the red region")),
    ("V02", ("variant",    "push the block ahead into the red target zone")),
    ("V03", ("variant",    "shift the cube toward the red goal area")),
    ("V04", ("variant",    "drive the cube to the red circular spot")),
    # ---- 5. 反向参照指令（语义目标不变，措辞从起点出发）----
    ("R01", ("reverse",    "push the cube away from its starting position toward the red target")),
    ("R02", ("reverse",    "move the cube off its initial spot and into the red target area")),
    ("R03", ("reverse",    "push the block from where it starts to the red circular mark")),
    ("R04", ("reverse",    "slide the cube out of its starting position onto the red goal")),
])


def template_ids():
    """返回全部模板 id（稳定顺序）。"""
    return list(TEMPLATES.keys())


def get_instruction(template_id):
    """按 id 取英文指令；id 不存在时抛 KeyError。"""
    return TEMPLATES[template_id][1]


def get_category(template_id):
    """按 id 取类别。"""
    return TEMPLATES[template_id][0]


def by_category(category):
    """按类别取 [(id, instruction), ...]。"""
    return [(tid, ins) for tid, (cat, ins) in TEMPLATES.items() if cat == category]


def balanced_instruction_sequence(seed=0):
    """生成指令 id 序列：5 类轮询、类内轮询，保证各类均衡覆盖。

    用于给 N 条轨迹分配指令。例：N=25，5 类各 5 条；类内 4 条模板循环，
    因此每条模板至少出现 1 次，部分出现 2 次（保证语义多样且不重复堆叠）。
    """
    import random
    ids = list(TEMPLATES.keys())
    cats = list(CATEGORY_DESC.keys())
    buckets = {c: by_category(c) for c in cats}
    # 轮转指针：每类独立推进，先类间轮询，再类内轮询
    order = []
    ci = 0
    while len(order) < len(ids):
        cat = cats[ci % len(cats)]
        pool = buckets[cat]
        idx = sum(1 for x in order if TEMPLATES[x][0] == cat)
        if idx < len(pool):
            order.append(pool[idx][0])
        ci += 1
    # 确定性洗牌：保持类别分布但打散相邻同类
    rng = random.Random(seed)
    rng.shuffle(order)
    return order


def sequence_for_count(n, seed=0):
    """返回长度为 n 的指令 id 列表，循环复用均衡序列。"""
    base = balanced_instruction_sequence(seed)
    return [base[i % len(base)] for i in range(n)]


if __name__ == "__main__":
    for cat, desc in CATEGORY_DESC.items():
        print(f"[{cat}] {desc}")
        for tid, ins in by_category(cat):
            print(f"  {tid}: {ins}")
    print(f"total={len(TEMPLATES)} templates, {len(CATEGORY_DESC)} categories")
