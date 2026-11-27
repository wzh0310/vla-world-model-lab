# VLA 微调统一工程门面（vla_finetune）

把零散的训练、评估、推理脚本整理为一个结构统一、可复用、符合开源规范的
Python 包。本目录是**统一入口**，底层实现仍在 `src/vla_finetune/` 与 `vla/`，
以子进程方式调用，因此历史实验（tag `v1.7`~`v1.9`）的结果完全可复现。

## 1. 目录结构

```
vla_finetune/
├── README.md          本文件
├── __init__.py        包入口与版本号
├── train.py           微调主入口（配置加载 / 参数统一 / 路径校验 / 异常处理）
├── eval.py            批量评估与泛化测试（闭环 / 泛化 / 跨任务 / 汇总）
├── inference.py       单样本推理
├── run.py             配置驱动一键启动
├── configs/           训练、LoRA、运行配置（见 configs/README.md）
├── data/              数据处理与格式转换脚本
└── utils/             动作归一化、可视化、运行时工具
```

## 2. 设计原则

1. **单一职责**：门面只做配置加载、参数统一、校验与调度；模型与数据逻辑留在底层。
2. **口径集中**：动作维度、归一化 key、LoRA 缩放等约定集中在 `configs/`，避免三处漂移。
3. **统一异常**：所有门面错误经 `utils.runtime.PipelineError` 收敛，输出一致的退出码。
4. **可干跑**：`--dry-run` 打印完整命令链而不执行，便于审阅与安全检查。
5. **向后兼容**：底层脚本与历史结果零改动，门面是新增的薄封装层。

## 3. 快速开始（容器 wzh-ai，仓库根，vla 环境）

```bash
PY=/opt/miniconda3/envs/vla/bin/python

# 1) 干跑校验：查看将要执行的底层命令
$PY vla_finetune/train.py --profile default --dry-run

# 2) 训练管线贯通验证
$PY vla_finetune/train.py --profile default --smoke

# 3) 完整训练，命令行覆盖参数
$PY vla_finetune/train.py --profile default --lora-rank 16 --learning-rate 0.0001

# 4) 配置驱动一键启动（rank 消融）
$PY vla_finetune/run.py --config vla_finetune/configs/run_ablation.yaml \
    --set args.lora_rank=16 --dry-run

# 5) 批量评估（闭环）
$PY vla_finetune/eval.py --suite closed \
    --adapter results/vla_finetune_tuned/best_adapter --max-steps 50

# 6) 单样本推理
$PY vla_finetune/inference.py --instruction "push the cube to the red target" --dry-run
```

## 4. 参数接口约定

- 所有门面参数使用连字符（如 `--lora-rank`），内部字典用下划线，转换由
  `utils.runtime.build_command` 统一完成；
- `--config` 直接指定 YAML 时优先级高于 `--profile`；
- `--dry-run` 在所有门面与 `run.py` 中语义一致：只打印不执行。

## 5. 与底层实现的关系

| 门面 | 底层脚本 | 说明 |
|---|---|---|
| `train.py` | `src/vla_finetune/train.py` | Trainer API QLoRA 实现 |
| `eval.py --suite closed` | `vla/eval/eval.py` | 闭环成功率评估 |
| `eval.py --suite generalization` | `vla/eval/eval_generalization_vla.py` | 指令/位置泛化 |
| `eval.py --suite cross` | `vla/eval/eval_cross_task.py` | 少样本跨任务 |
| `inference.py` | `vla/inference/inference.py` | 官方 prompt + 动作映射 |
| `run.py` | 上述门面 | 配置驱动分发 |

## 6. 异常处理

门面在以下情况返回退出码 2 并在 stderr 打印 `[门面错误]`：

- 配置文件不存在 / YAML 解析失败 / 缺少必需字段；
- 模型、数据集、统计量等关键路径不存在（`--dry-run` 时跳过路径校验）；
- 底层脚本非零退出 / 解释器缺失。

原始底层脚本的 stdout/stderr 原样透传，便于排查训练细节。
