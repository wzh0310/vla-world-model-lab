# -*- coding: utf-8 -*-
"""批量评估与泛化测试统一入口（门面）。

把四类评估统一到一个 CLI，底层调用各专属脚本（保持实现不变）：

- closed:       基础闭环评估       -> vla/eval/eval.py
- generalization: 指令/位置泛化   -> vla/eval/eval_generalization_vla.py
- cross:        少样本跨任务       -> vla/eval/eval_cross_task.py
- summarize:    结果汇总与图表     -> vla/eval/summarize_cross.py 等

用法（容器 wzh-ai，仓库根，vla 环境）：
    python vla_finetune/eval.py --suite closed --adapter results/vla_finetune_tuned/best_adapter --dry-run
    python vla_finetune/eval.py --suite generalization --adapter <path> --scenes <json> --out <json>
    python vla_finetune/eval.py --suite cross --algo vla --adapter <path> --scenes <json> --out <json>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vla_finetune.utils.runtime import PipelineError, emit_error, run_script  # noqa: E402

BACKENDS = {
    "closed": "vla/eval/eval.py",
    "generalization": "vla/eval/eval_generalization_vla.py",
    "cross": "vla/eval/eval_cross_task.py",
    "summarize-cross": "vla/eval/summarize_cross.py",
    "summarize-generalization": "vla/eval/summarize_generalization.py",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="VLA 批量评估与泛化测试统一入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--suite", required=True, choices=sorted(BACKENDS),
                    help="评估套件")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    # 闭环评估参数
    ap.add_argument("--mode", choices=("finetuned", "zero-shot"), default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--instructions", nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--seed-base", type=int, default=None)
    # 泛化 / 跨任务参数
    ap.add_argument("--scenes", default=None, help="场景 JSON 路径")
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    ap.add_argument("--algo", choices=("vla", "act"), default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes-per-task", type=int, default=None)
    ap.add_argument("--train-dataset", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        backend = BACKENDS[args.suite]
        # suite 专属参数白名单：避免把无关参数透传给底层脚本
        common = {"adapter": args.adapter, "max_steps": args.max_steps,
                  "out": args.out, "scenes": args.scenes}
        by_suite = {
            "closed": {
                "mode": args.mode, "model_dir": args.model_dir,
                "instructions": args.instructions, "episodes": args.episodes,
                "scale": args.scale, "seed_base": args.seed_base,
            },
            "generalization": {"train_dataset": args.train_dataset},
            "cross": {
                "algo": args.algo, "checkpoint": args.checkpoint,
                "episodes_per_task": args.episodes_per_task,
            },
            "summarize-cross": {},
            "summarize-generalization": {},
        }
        backend_args = dict(common)
        backend_args.update(by_suite[args.suite])
        # 必填参数检查（仅在实际执行时）
        if not args.dry_run:
            if args.suite in ("generalization", "cross") and not args.scenes:
                raise PipelineError(f"--suite {args.suite} 需要 --scenes")
            if args.suite in ("generalization", "cross", "closed") and not args.out \
                    and args.suite != "closed":
                raise PipelineError(f"--suite {args.suite} 需要 --out")
            if args.suite == "cross" and not args.algo:
                raise PipelineError("--suite cross 需要 --algo vla|act")
        return run_script(backend, backend_args, dry_run=args.dry_run)
    except PipelineError as exc:
        return emit_error(exc)


if __name__ == "__main__":
    sys.exit(main())
