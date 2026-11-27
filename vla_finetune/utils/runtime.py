# -*- coding: utf-8 -*-
"""工程化运行时工具：子进程调度、配置加载、路径校验与统一异常处理。

本模块是统一门面（vla_finetune/）与底层实现（src/vla_finetune/、vla/）之间
的唯一桥梁。底层脚本保持原样，门面通过子进程调用它们，因此：

- 历史实验（tag v1.7~v1.9）对应的底层代码零改动，结果可复现；
- 门面层集中处理配置、校验、异常，避免把工程逻辑散落到各训练/评估脚本。

所有函数以「返回退出码」或「抛出 PipelineError」的方式给出明确结果，
便于 cli.py 转换为进程退出码。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

# 仓库根目录：本文件位于 <ROOT>/vla_finetune/utils/runtime.py
ROOT = Path(__file__).resolve().parents[2]

# 底层脚本统一使用 vla conda 环境的解释器；可用环境变量覆盖（便于测试）。
VLA_PYTHON = os.environ.get("VLA_PYTHON", "/opt/miniconda3/envs/vla/bin/python")


class PipelineError(RuntimeError):
    """门面层统一异常：配置错误、路径缺失或子进程执行失败。"""


def load_config(path: str | Path, *, flatten: bool = True) -> dict:
    """加载 YAML 配置，附带友好异常。

    与底层 src/vla_finetune/train.py 的 load_yaml 口径一致：
    flatten=True（默认）时，值为字典的分节会被展开为平铺键，供训练配置使用；
    flatten=False 时保留原始嵌套结构，供 run.py 的一键运行配置使用（它依赖
    顶层的 args 字典做点分路径覆盖）。
    """
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise PipelineError(f"配置文件不存在: {p}")
    try:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise PipelineError(f"配置文件解析失败 {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise PipelineError(f"配置文件顶层必须是字典: {p}")
    if not flatten:
        return data
    flat: dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(value)  # 分节展开
        else:
            flat[key] = value
    return flat


def require_paths(*paths: str | Path) -> None:
    """校验必需路径（模型 / 数据集 / 统计量）存在，缺失则抛出 PipelineError。"""
    missing = [str(p) for p in paths if p is not None and not Path(p).exists()]
    if missing:
        raise PipelineError("必需路径不存在:\n  - " + "\n  - ".join(missing))


def build_command(script_rel: str, args: dict, *, python: str | None = None) -> list:
    """把 (参数名 -> 值) 字典转换为底层脚本的命令行参数列表。

    约定：
    - 值为 None 的参数被忽略；
    - 布尔 True 追加为独立开关 --name，False 忽略；
    - 其余值接受 --name value 形式；参数名传入时不带前导 --。
    """
    script = Path(script_rel)
    if not script.is_absolute():
        script = ROOT / script
    cmd = [python or VLA_PYTHON, str(script)]
    for key, value in args.items():
        if value is None:
            continue
        # 门面内部用下划线（如 batch_size），底层 argparse 用连字符（--batch-size）
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def run_script(
    script_rel: str,
    args: dict,
    *,
    dry_run: bool = False,
    env_extra: dict | None = None,
    python: str | None = None,
) -> int:
    """以子进程执行仓库内脚本，返回退出码。

    dry_run=True 时只打印命令不执行（用于门面自检）；
    子进程非零退出会抛出 PipelineError，附带脚本名与退出码。
    """
    cmd = build_command(script_rel, args, python=python)
    pretty = " ".join(str(c) for c in cmd)
    print(f"[门面] 执行: {pretty}", flush=True)
    if dry_run:
        print("[门面] dry-run 模式，未实际执行。", flush=True)
        return 0

    script_path = Path(script_rel)
    if not script_path.is_absolute():
        script_path = ROOT / script_path
    if not script_path.exists():
        raise PipelineError(f"底层脚本不存在: {script_path}")
    if python is None and not Path(VLA_PYTHON).exists():
        raise PipelineError(
            f"vla 环境解释器不存在: {VLA_PYTHON}（可用 VLA_PYTHON 环境变量覆盖）")

    env = os.environ.copy()
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})

    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    except OSError as exc:
        raise PipelineError(f"子进程启动失败: {exc}") from exc
    if proc.returncode != 0:
        raise PipelineError(
            f"底层脚本执行失败（退出码 {proc.returncode}）: {script_rel}")
    return proc.returncode


def emit_error(exc: Exception) -> int:
    """统一打印异常到 stderr 并返回退出码 2。"""
    print(f"[门面错误] {exc}", file=sys.stderr, flush=True)
    return 2
