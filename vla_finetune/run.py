# -*- coding: utf-8 -*-
"""一键启动脚本（配置驱动）。

从一个 YAML 运行配置读取完整流程参数，支持通过 --set 在命令行覆盖任意字段，
再分发到 train / eval / infer 门面。适合用配置文件快速切换训练参数，
避免反复手写长命令。

运行配置格式（见 configs/run_*.yaml）：
    task: train|eval|infer          # 要执行的门面
    entry: ...                      # 说明性字段
    args:                           # 传给对应门面的参数字典
        profile: ablation
        lora_rank: 8
        ...

用法（容器 wzh-ai，仓库根，vla 环境）：
    python vla_finetune/run.py --config vla_finetune/configs/run_smoke.yaml --dry-run
    python vla_finetune/run.py --config vla_finetune/configs/run_ablation.yaml --set lora_rank=16
    python vla_finetune/run.py --config vla_finetune/configs/run_smoke.yaml --set task=train --set args.smoke=true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vla_finetune.utils.runtime import PipelineError, emit_error, load_config, run_script  # noqa: E402

FACADE = {
    "train": "vla_finetune/train.py",
    "eval": "vla_finetune/eval.py",
    "infer": "vla_finetune/inference.py",
}


def _coerce(raw: str):
    """把 --set 的字符串值转换为 bool / int / float / str。"""
    low = raw.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_dot_set(config: dict, key_path: str, value) -> None:
    """按点分路径覆盖嵌套配置，如 args.lora_rank=16。"""
    parts = key_path.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise PipelineError(f"--set 路径不存在: {key_path}")
        node = node[part]
    node[parts[-1]] = value


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="配置驱动的一键启动脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", required=True, help="运行配置 YAML 路径")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖配置字段（可多次，支持点分路径）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        # 一键运行配置保留嵌套结构（args 字典），不做分节展开
        cfg = load_config(args.config, flatten=False)
        for item in args.set:
            if "=" not in item:
                raise PipelineError(f"--set 需要 KEY=VALUE 形式: {item}")
            key, raw = item.split("=", 1)
            _apply_dot_set(cfg, key.strip(), _coerce(raw.strip()))

        task = cfg.get("task")
        if task not in FACADE:
            raise PipelineError(f"运行配置 task 必须是 {sorted(FACADE)}，当前={task!r}")
        facade_args = cfg.get("args") or {}
        if not isinstance(facade_args, dict):
            raise PipelineError("运行配置 args 必须是字典")

        # 一键脚本只负责分发；门面自身还会做二次校验。
        # dry-run 作为门面参数透传：门面收到后打印底层命令且不执行，
        # 这样整条命令链可见（run -> 门面 -> 底层）。
        forwarded = dict(facade_args)
        if args.dry_run:
            forwarded["dry_run"] = True
        return run_script(FACADE[task], forwarded, dry_run=False)
    except PipelineError as exc:
        return emit_error(exc)


if __name__ == "__main__":
    sys.exit(main())
