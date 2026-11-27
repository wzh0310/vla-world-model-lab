# 统一配置目录

本目录集中 VLA 微调与评估的全部配置，按用途分三类：

## 1. 训练配置（供 `train.py --profile` 使用）

| 文件 | 用途 |
|---|---|
| `default.yaml` | 默认 QLoRA 微调（rank=8, lr=2e-4） |
| `ablation.yaml` | rank 消融（固定 192 步 / lr=1e-4 / alpha=2r） |
| `fewshot.yaml` | 跨任务少样本续训（lr=2e-5, constant, 60 步） |

配置文件为分节结构（paths / action / lora / optim / runtime），底层 `load_yaml`
会自动展开为平铺键。任意字段均可由命令行覆盖，例如：

```bash
python vla_finetune/train.py --profile ablation --lora-rank 16 --lr-scheduler constant
```

## 2. 一键运行配置（供 `run.py --config` 使用）

| 文件 | 用途 |
|---|---|
| `run_smoke.yaml` | 训练管线贯通验证 |
| `run_ablation.yaml` | rank 消融（可用 `--set args.lora_rank=16` 切换） |
| `run_eval.yaml` | 闭环评估（可用 `--set args.adapter=<path>` 指定权重） |

`run.py` 读取 `task` 决定分发到哪个门面，`args` 作为该门面的参数字典。
`--set KEY=VALUE` 支持点分路径覆盖嵌套字段，值自动推断为 bool/int/float/str。

```bash
python vla_finetune/run.py --config vla_finetune/configs/run_ablation.yaml \
    --set args.lora_rank=16 --dry-run
```

## 3. 口径约定（改动前务必确认）

- `unnorm_key: pushcube_vla`：本项目注册的动作统计量 key，**禁止**沿用 `bridge_orig`；
- `action_dim: 7` / `source_action_dim: 4`：数据集 4 维扩为 OpenVLA 7 维；
- `alpha = 2 * rank`：消融时保持 LoRA 缩放恒定，确保只比较秩的影响；
- 消融固定预算：192 optimizer 步、lr=1e-4、seed=42。
