# -*- coding: utf-8 -*-
"""微调统一入口（门面）。

职责：加载统一配置 -> 命令行覆盖 -> 校验路径 -> 调用底层
src/vla_finetune/train.py。底层实现保持不变，历史实验可复现。

用法（容器 wzh-ai，仓库根，vla 环境）：
    # 干跑：只打印将执行的命令
    python vla_finetune/train.py --profile default --dry-run
    # 贯通验证
    python vla_finetune/train.py --profile default --smoke
    # 完整训练，命令行覆盖 rank / 学习率
    python vla_finetune/train.py --profile default --lora-rank 16 --learning-rate 0.0001
    # LoRA rank 消融预设（固定预算）
    python vla_finetune/train.py --profile ablation --lora-rank 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 仓库根加入 sys.path，保证门面包可独立运行
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vla_finetune.utils.runtime import (  # noqa: E402
    PipelineError,
    emit_error,
    load_config,
    require_paths,
    run_script,
)

CONFIGS = _ROOT / "vla_finetune" / "configs"
BACKEND = "src/vla_finetune/train.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="OpenVLA QLoRA 微调统一入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--profile", default="default",
                    help="配置预设名，对应 configs/<profile>.yaml")
    ap.add_argument("--config", default=None,
                    help="直接使用指定 YAML 配置（覆盖 --profile）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    ap.add_argument("--smoke", action="store_true", help="1%% 数据贯通验证")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--max-grad-norm", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr-scheduler", default=None)
    ap.add_argument("--init-adapter", default=None, help="从已有 adapter 续训")
    ap.add_argument("--hf-dataset", default=None)
    ap.add_argument("--eval-subset", type=int, default=None)
    ap.add_argument("--eval-steps", type=int, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cfg_path = args.config or (CONFIGS / f"{args.profile}.yaml")
        flat = load_config(cfg_path)
        # 把统一配置写回一个临时扁平 YAML 供底层 --config 使用：
        # 直接把分节原文件传给底层即可（底层 load_yaml 同样展开分节），
        # 这里仅校验关键字段存在。
        required_keys = {"model_dir", "hf_dataset", "norm_stats", "out_dir",
                         "action_dim", "unnorm_key", "lora_rank", "lora_alpha"}
        missing_keys = sorted(required_keys - set(flat))
        if missing_keys:
            raise PipelineError(f"配置缺少必需字段: {missing_keys}（文件: {cfg_path}）")

        if not args.dry_run:
            require_paths(flat["model_dir"], flat["hf_dataset"], flat["norm_stats"])

        backend_args = {
            "config": str(Path(cfg_path).relative_to(_ROOT)) if Path(cfg_path).is_absolute()
            else str(cfg_path),
            "smoke": args.smoke,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
            "out_dir": args.out_dir,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "max_steps": args.max_steps,
            "lr_scheduler": args.lr_scheduler,
            "init_adapter": args.init_adapter,
            "hf_dataset": args.hf_dataset,
            "eval_subset": args.eval_subset,
            "eval_steps": args.eval_steps,
        }
        return run_script(BACKEND, backend_args, dry_run=args.dry_run)
    except PipelineError as exc:
        return emit_error(exc)


if __name__ == "__main__":
    sys.exit(main())
