# -*- coding: utf-8 -*-
"""可视化工具门面。

统一封装已有绘图脚本的调用方式，供报告与作品集复用。底层绘图脚本位于
results/ 各实验目录与 src/vla_ablation/summarize.py，这里只提供统一的
子进程入口与输出路径校验，不改动绘图逻辑本身。
"""

from __future__ import annotations

from pathlib import Path

from .runtime import ROOT, PipelineError, run_script


def plot_success_bars(summary_json: str, out_png: str, script_rel: str | None = None) -> int:
    """调用指定绘图脚本生成成功率柱状图。

    script_rel 默认指向跨任务绘图脚本；泛化类图可显式传入对应脚本路径。
    """
    script = script_rel or "vla/eval/plot_cross.py"
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 绘图脚本大多通过读取固定 summary 工作，参数不一定支持 --out；
    # 这里仅做存在性校验，具体参数以底层脚本为准。
    if not (Path(summary_json).exists() or (ROOT / summary_json).exists()):
        raise PipelineError(f"汇总 JSON 不存在: {summary_json}")
    return run_script(script, {}, dry_run=False)


def list_plot_scripts() -> list:
    """列出仓库内可复用的绘图脚本（相对仓库根路径）。"""
    candidates = [
        "vla/eval/plot_cross.py",
        "vla/eval/plot_generalization.py",
        "src/vla_ablation/summarize.py",
        "scripts/plot_closed_loop.py",
    ]
    return [p for p in candidates if (ROOT / p).exists()]
