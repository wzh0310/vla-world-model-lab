# Trainer 版 QLoRA 微调脚本开发

本文档记录基于 `transformers.Trainer` 的 OpenVLA 微调主脚本、动作归一化工具、数据管线适配，以及小样本试跑的真实结果。所有数字来自容器 `wzh-ai` 的 A6000 实测，未使用估计值。小样本只验证管线与 loss 下降趋势，**不构成收敛或任务成功率结论**。

## 1. 脚本与配置

| 文件 | 作用 |
| --- | --- |
| `src/vla_finetune/train.py` | 微调主脚本（Trainer API + 4bit + LoRA） |
| `src/vla_finetune/action_utils.py` | 动作归一化/反归一化、4D↔7D 互转、动作 token 编解码 |
| `src/vla_finetune/trainer_data.py` | HF Dataset → PyTorch Dataset + collator，prompt 严格对齐官方 |
| `vla/configs/trainer_lora.yaml` | 手册指定超参的配置 |

配置严格采用手册数值：LoRA `rank=8 / alpha=16 / dropout=0.05`、目标 `q_proj/v_proj`；`lr=2e-4`、`batch_size=4`、`grad_accum=4`（等效 batch 16）、`epochs=10`、`warmup_steps=50`；4bit NF4 量化加载。

## 2. 关键实现与适配

### 2.1 模型加载与冻结

- 用 `BitsAndBytesConfig(load_in_4bit=True, nf4, double_quant, compute_dtype=bf16)` 加载，`device_map` 显式落到单卡；量化模型禁止再调用 `.to()`。
- 冻结底座全部参数，只对语言主干 `self_attn.q_proj / v_proj` 挂 LoRA（可训练 4,194,304 参数，约底座的 0.107%）。视觉编码器 DINOv2/SigLIP 与 projector 全部冻结。
- 关于"交叉注意力层"：OpenVLA（Prismatic）把图像 patch token 与文本 token **拼接后统一送入语言主干的 self-attention**，架构中没有独立的 `cross_attn` 模块。因此手册指定的 `q_proj/v_proj` 正是图文交叉发生的注意力投影；本实现严格按手册目标模块执行，不虚构不存在的模块名。
- `TrainingArguments` 关闭 bf16/fp16 autocast：4bit 模型视觉塔权重为 fp16，自定义 `QLoRATrainer.compute_loss` 在送入模型前把 `pixel_values` 转为 fp16，避免 dtype 冲突。

### 2.2 prompt 模板严格对齐

`trainer_data.py` 直接按文件路径加载 `vla/utils/vla_adapter.py::build_prompt`，使用官方模板：

```
In: What action should the robot take to {instruction}?\nOut:
```

训练与推理共用同一函数，避免模板漂移导致微调失效。实测样本 prompt 与模板一致。

### 2.3 动作归一化 / 反归一化管线

`action_utils.py` 提供并集成到数据处理管线：

- `ActionNorm.normalize / denormalize`：min-max ↔ [-1,1]，常量维（gripper span=0）单独处理，roundtrip 误差 0.0；
- `expand_4d_to_7d / squeeze_7d_to_4d`：数据集 4 维 `[dx,dy,dz,gripper]` 扩成 OpenVLA 7 维（旋转三维置 0）；
- `ActionTokenizer`：连续动作 ↔ 离散动作 token（词表尾部 `vocab_size-1-bin_id`），roundtrip 最大误差 0.00392（与 256 bin 量化步长一致）。

训练侧把归一化 4 维扩成 7 维再 token 化，labels 只对动作位监督；注册 `unnorm_key=pushcube_vla` 到模型配置，推理侧可反归一化回物理空间。

## 3. 小样本试跑（真实结果）

配置：1% 训练数据 = 40 帧（不少于 1 个完整累积 batch）、20 个 optimizer 步、`warmup_steps` 临时覆盖为 2（完整训练仍为 50，因 20 步内 warmup=50 无法完成升温、看不出 loss 趋势）、其余超参同配置。

| 指标 | 值 |
| --- | --- |
| train loss 首步 → 末步 | **12.8332 → 2.4911** |
| 最小 loss | 1.7700（第 19 步） |
| eval loss | 7.5832 |
| 可训练参数 | 4,194,304（4bit 底座 3.903B） |
| 加载显存 → 训练峰值 | 4.551 GB → 12.01 GB |
| 耗时 | 110.54 s（约 0.18 step/s） |

loss 序列（20 步）呈清晰下降趋势：12.83 → 12.79 → 12.50 → 12.11 → 10.49 → 9.24 → 8.25 → 7.27 → 6.25 → 5.38 → 4.42 → 4.77 → 3.11 → 3.61 → 2.85 → 2.71 → 2.35 → 2.46 → 1.77 → 2.49。

**判读边界**：

- 管线贯通：模型加载、4bit LoRA、prompt、动作 token 化、反向传播、评估、adapter 保存全部无报错；
- loss 下降只证明优化方向正确；1% 数据下 eval loss 明显高于 train loss 属小样本过拟合预期，不代表泛化；
- 训练峰值显存 12 GB 是反向传播激活所致，与推理 4bit ≤8 GB 不是同一口径；
- 本轮**未跑完整 10 epoch 全量训练**，不宣称收敛或成功率。

结果文件：`results/vla_finetune_trainer/trainer_smoke_result.json`、`finetune_meta.json`。大权重 adapter 与 tokenizer 已在 `.gitignore` 中排除，不入库。

## 4. 复现命令（vla 环境）

```bash
cd /data/wzh/robot/vla-world-model-lab
# 小样本贯通（观察 loss 下降）
/opt/miniconda3/envs/vla/bin/python src/vla_finetune/train.py \
  --config vla/configs/trainer_lora.yaml --smoke --smoke-warmup 2 --smoke-steps 20

# 完整训练（手册超参，warmup=50）
/opt/miniconda3/envs/vla/bin/python src/vla_finetune/train.py \
  --config vla/configs/trainer_lora.yaml
```
