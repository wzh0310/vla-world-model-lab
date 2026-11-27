# -*- coding: utf-8 -*-
"""动作归一化工具门面。

动作口径（4D 数据集 <-> 7D OpenVLA）是训练/推理/环境三处最容易漂移的地方，
因此这里不重新实现，而是显式复用底层 src/vla_finetune/action_utils.py，
保证门面与底层训练使用同一份代码。

暴露：
- ActionNorm: 4D 物理量 <-> [-1,1] 归一化
- ActionTokenizer: 7D 动作 <-> token id
- roundtrip_error: 编解码一致性自检
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# 仓库根：本文件位于 <ROOT>/vla_finetune/utils/action.py
ROOT = Path(__file__).resolve().parents[2]
_IMPL = ROOT / "src" / "vla_finetune" / "action_utils.py"

if not _IMPL.exists():
    raise ImportError(f"底层动作工具不存在: {_IMPL}")

_spec = importlib.util.spec_from_file_location("_vf_action_utils_impl", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)  # type: ignore[union-attr]

ActionNorm = _impl.ActionNorm
ActionTokenizer = _impl.ActionTokenizer
roundtrip_error = _impl.roundtrip_error

__all__ = ["ActionNorm", "ActionTokenizer", "roundtrip_error"]
