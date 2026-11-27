# -*- coding: utf-8 -*-
"""单样本推理统一入口（门面）。

底层调用 vla/inference/inference.py（官方 prompt + 自定义动作映射口径不变），
门面统一参数命名并增加异常提示。

用法（容器 wzh-ai，仓库根，vla 环境）：
    python vla_finetune/inference.py --instruction "push the cube to the red target" --dry-run
    python vla_finetune/inference.py --image <npz> --instruction "..." --out <json>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vla_finetune.utils.runtime import PipelineError, emit_error, run_script  # noqa: E402

BACKEND = "vla/inference/inference.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="OpenVLA 单样本推理统一入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--quant", choices=("int4", "none"), default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--image", default=None, help="观测图像/npz 路径")
    ap.add_argument("--state", default=None)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-env-step", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        backend_args = {
            "model_dir": args.model_dir, "quant": args.quant, "device": args.device,
            "unnorm_key": args.unnorm_key, "image": args.image, "state": args.state,
            "instruction": args.instruction, "scale": args.scale, "seed": args.seed,
            "out": args.out, "no_env_step": args.no_env_step,
        }
        return run_script(BACKEND, backend_args, dry_run=args.dry_run)
    except PipelineError as exc:
        return emit_error(exc)


if __name__ == "__main__":
    sys.exit(main())
