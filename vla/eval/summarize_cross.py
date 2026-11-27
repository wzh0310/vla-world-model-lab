# -*- coding: utf-8 -*-
"""汇总跨任务与横向对比：VLA 基线 / VLA few-shot / ACT，生成对比表与 JSON。"""
import json
from pathlib import Path

BASE = Path("results/vla_cross_task")
GEN = Path("results/vla_generalization")


def load(p):
    return json.load(open(p, encoding="utf-8"))


def rate(s):
    return f"{s['success_count']}/{s['n_episodes']} ({s['success_rate']:.2f})"


def main():
    base = load(BASE / "vla_baseline_raw.json")
    few = load(BASE / "vla_fewshot_raw.json")
    act = load(BASE / "act_raw.json")
    gc = load(GEN / "GENERALIZATION_COMPARISON.json")

    lines = [
        "# 少样本跨任务泛化与完整横向对比结果",
        "",
        "## 一、跨任务（推到左边 / 右边）—— 5 条示教 few-shot 适配",
        "",
        "| 策略 | 左边 | 右边 | 总体 | 平均距离改善(m) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, d in [("VLA 基线（未适配）", base),
                    ("VLA few-shot（5条示教）", few),
                    ("ACT（同场景对照）", act)]:
        ov = d["summary"]["overall"]
        lines.append(f"| {name} | {rate(d['summary']['left'])} | "
                     f"{rate(d['summary']['right'])} | {rate(ov)} | "
                     f"{ov['mean_dist_improvement']:+.4f} |")

    lines += [
        "",
        "few-shot 设置：左 2 条 / 右 3 条，共 5 条真实几何示教（177 帧）；"
        "从最优 adapter 续训，lr=2e-5、constant、60 步；"
        "val 用原任务子集监控遗忘。",
        "",
        "## 二、四维度横向对比（含样本效率）",
        "",
        "| 维度 | VLA | ACT | 口径说明 |",
        "| --- | --- | --- | --- |",
        f"| 见过的任务成功率 | 0.20 (3/15) | {rate(gc['act']['instruction'])} | "
        "VLA 取基础闭环评估；ACT 取泛化框架的固定场景重复（其 instruction 类无语言输入） |",
        f"| 新指令泛化 | {rate(gc['vla']['instruction'])} | {rate(gc['act']['instruction'])} | "
        "VLA 为中文未见指令（语种级 OOD）；ACT 无语言输入，仅固定场景重复 |",
        f"| 新初始位置泛化 | {rate(gc['vla']['start_pos'])} | {rate(gc['act']['start_pos'])} | "
        "起点 ±5cm 随机 20 次 |",
        f"| 新目标位置泛化 | {rate(gc['vla']['target_pos'])} | {rate(gc['act']['target_pos'])} | "
        "目标前后左右偏移 20 次 |",
        f"| 样本效率（跨任务 5 条示教） | {rate(few['summary']['overall'])} | "
        f"{rate(act['summary']['overall'])} | "
        "VLA 仅 5 条新示教适配；ACT 为该场景整数据训练 checkpoint，未做同规模 few-shot |",
        "",
        "## 三、机器可读汇总",
        "",
        "三算法跨任务原始结果：`vla_baseline_raw.json`、`vla_fewshot_raw.json`、`act_raw.json`；"
        "泛化四维度原始结果见 `results/vla_generalization/GENERALIZATION_COMPARISON.json`。",
        "",
    ]
    out = BASE / "comparison_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print("SUMMARIZE CROSS DONE")


if __name__ == "__main__":
    main()
