# -*- coding: utf-8 -*-
"""汇总 VLA 与 ACT 的三类泛化测试，生成量化对比表格与中文报告。"""
import json
from pathlib import Path

RAW = Path("results/vla_generalization")
TESTS = [("instruction", "指令泛化"), ("start_pos", "初始位置泛化"),
         ("target_pos", "目标位置泛化"), ("overall", "总体")]


def load(p):
    return json.loads((RAW / p).read_text(encoding="utf-8"))


def main():
    vla = load("VLA_generalization_raw.json")
    act = load("ACT_generalization_raw.json")
    scene = load("scenes.json") if (RAW / "scenes.json").exists() else None

    lines = ["# VLA 与 ACT 泛化能力量化对比", "",
             "同一批场景文件、同一成功判据（立方体中心距目标 < 0.03 m）、同一 50 步上限；",
             "VLA 使用微调最优 adapter（int4 + LoRA），ACT 使用 `act_chunk40` checkpoint。", "",
             "> 说明：ACT 配置含 13 维状态观测，并非纯视觉；且 ACT 无语言条件，",
             "> 其「指令泛化」一类是固定场景重复，不代表语言泛化，仅作同协议对照。", "",
             "## 总体对比", "",
             "| 测试类别 | VLA 成功率 | VLA 平均距离改善 | ACT 成功率 | ACT 平均距离改善 |",
             "| --- | --- | --- | --- | --- |"]
    for key, name in TESTS:
        vs, as_ = vla["summary"][key], act["summary"][key]
        lines.append(f"| {name} | {vs['success_count']}/{vs['n_episodes']} = {vs['success_rate']} "
                     f"| {vs['mean_dist_improvement']} m | {as_['success_count']}/{as_['n_episodes']} = {as_['success_rate']} "
                     f"| {as_['mean_dist_improvement']} m |")

    lines += ["", "## 指令泛化：10 条训练集未见同义指令", "",
              "训练集共 %d 条指令，10 条测试指令与训练集字符串重合数 = %d（零重合已校验）。"
              % (vla["protocol"]["train_instruction_count"],
                 len(vla["protocol"]["novel_overlap_with_train"])), "",
              "| 序号 | 指令 | VLA 成功 | VLA 平均距离改善 |", "| --- | --- | --- | --- |"]
    novel = vla["protocol"]["novel_instructions"]
    ins_rec = vla["per_test"]["instruction"]
    for i, ins in enumerate(novel):
        rows = [r for r in ins_rec if r["instruction"] == ins]
        sc = sum(r["success"] for r in rows)
        imp = sum(r["dist_improvement"] for r in rows) / max(len(rows), 1)
        lines.append(f"| {i+1} | {ins} | {sc}/{len(rows)} | {imp:+.4f} m |")

    # 目标方向分解
    lines += ["", "## 目标位置泛化：按方向分解（VLA）", "",
              "| 方向 | 成功 | 平均距离改善 |", "| --- | --- | --- |"]
    for d in ("right", "left", "back", "front"):
        tr = [r for r in vla["per_test"]["target_pos"]
              if (r.get("direction") or scene and next(
                  (s.get("direction") for s in scene["scenes"]
                   if s["kind"] == "target_pos" and s["scene_id"] == r["scene_id"]), None)) == d]
        if not tr and scene:
            idmap = {s["scene_id"]: s.get("direction") for s in scene["scenes"]
                     if s["kind"] == "target_pos"}
            tr = [r for r in vla["per_test"]["target_pos"] if idmap.get(r["scene_id"]) == d]
        sc = sum(r["success"] for r in tr)
        imp = sum(r["dist_improvement"] for r in tr) / max(len(tr), 1)
        lines.append(f"| {d} | {sc}/{len(tr)} | {imp:+.4f} m |")

    (RAW / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "experiment": "VLA vs ACT generalization",
        "vla": {k: vla["summary"][k] for k, _ in TESTS},
        "act": {k: act["summary"][k] for k, _ in TESTS},
        "novel_instructions": novel,
        "novel_overlap_with_train": vla["protocol"]["novel_overlap_with_train"],
    }
    (RAW / "GENERALIZATION_COMPARISON.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary ->", RAW / "comparison_table.md")
    for key, name in TESTS:
        print(f"  {name}: VLA {vla['summary'][key]['success_rate']} "
              f"ACT {act['summary'][key]['success_rate']}")


if __name__ == "__main__":
    main()
