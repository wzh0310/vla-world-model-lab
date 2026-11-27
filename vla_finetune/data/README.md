# VLA 数据处理与格式转换

本目录集中数据处理脚本，供统一门面在训练前准备数据使用：

| 脚本 | 用途 |
|---|---|
| `build_hf_dataset.py` | 从原始 npz 构建 HuggingFace Dataset（train/val 分目录） |
| `hf_dataset_adapter.py` | HF Dataset 与训练字段的适配层 |
| `collect_variants.py` | 采集指令变体数据 |
| `instruction_variants.py` | 指令模板与变体定义 |
| `verify_adapter.py` / `verify_hf_dataset.py` | 数据格式与统计量一致性校验 |

数据与动作口径的统一约定见 `vla_finetune/configs/README.md` 与
`docs/LoRA微调配置与数据集字段规范.md`。训练侧通过统一配置中的
`hf_dataset` / `norm_stats` 字段引用处理后的数据，不在此目录硬编码路径。
