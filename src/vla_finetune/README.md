# OpenVLA LoRA 微调骨架（vla_finetune）

基于 `peft` 的 OpenVLA 7B LoRA 微调代码骨架，面向 PushCube v1.0 数据集。当前阶段目标是**跑通模型 / 数据集 / 训练代码链路**，不承担成功率指标；完整训练与效果评估留待下一阶段。

## 1. 目录结构

```
src/vla_finetune/
├── __init__.py       # 模块导出
├── config.py         # dataclass 配置：YAML 加载 + 命令行覆盖
├── data.py           # 动作 token 化、4D→7D 扩维、批次整理
├── model.py          # 模型加载、LoRA 挂载、unnorm_key 注册、元信息落盘
├── trainer.py        # 训练循环、验证评估、adapter 保存
├── train_lora.py     # CLI 入口（--smoke / 完整训练）
└── README.md

vla/configs/lora_finetune.yaml   # 全部超参数与路径
```

## 2. 设计要点

### 2.1 LoRA 配置

| 项 | 值 | 依据 |
|---|---|---|
| 挂载位置 | LLM 主干 `language_model.model.layers.N` 的 q/k/v/o + gate/up/down | 视觉塔 DINOv2 + SigLIP 冻结，只调语言→动作映射 |
| `lora_rank` / `alpha` | 32 / 16 | 官方默认；rank 对性能影响小，16 也可 |
| `lora_dropout` | 0.0 | 官方默认 |
| 可训练参数 | 约 1.4% | LoRA 足以匹敌全参数微调 |
| 量化 | **不量化**（bf16） | A6000 48GB 足够；QLoRA 实测掉约 16 点 |
| 梯度检查点 | 开 | 激活占训练显存约 53%，是最大开销 |

`target_modules` 用正则精确匹配 LLM 线性层，不碰 `vision_backbone.*` 与 `projector.*`。

### 2.2 动作空间严格对齐（最易出错）

数据集 4 维 `[dx,dy,dz,gripper]` → OpenVLA 7 维：

```
[dx, dy, dz, g]  ->  [dx, dy, dz, 0, 0, 0, g]
```

- 旋转三维补 0（推物任务无需旋转）；
- 数据已在 `v1.0/norm_stats.json` 中按 min-max 归一化到 `[-1,1]`；
- 动作 token 编号与模型 `predict_action` **互逆**：
  `token_id = vocab_size - 1 - argmin|bin_center - a|`，`vocab_size = text_config.vocab_size - pad_to_multiple_of = 32000`，`n_action_bins = 256`；
- `loss` 只对 7 个动作 token 位计算：prompt 位 label 置 `-100`，图像 patch 位由模型 forward 内部置 `-100`。

### 2.3 统计量不可混用

`v1.0/norm_stats.json` 是自采数据统计量，`register_custom_norm_stats()` 把它注册为模型 `norm_stats["pushcube_vla"]`：

```
q01 = min(4D) 扩 7D，q99 = max(4D) 扩 7D，旋转维 mask=False
```

因为 OpenVLA 反演 `0.5*(a+1)*(q99-q01)+q01` 与 min-max 反演 `(a+1)*(max-min)/2+min` 等价。**推理时必须用 `unnorm_key=pushcube_vla`**，绝不可沿用 `bridge_org`，否则动作尺度全错。adapter 同目录的 `finetune_meta.json` 保存了 key、动作维度、词表口径与统计量副本。

## 3. 环境与方法

在容器 `wzh-ai`、仓库根 `/data/wzh/robot/vla-world-model-lab` 执行，使用 `vla` 环境（`torch 2.6.0+cu124` / `transformers 4.40.1` / `peft 0.11.1` / `accelerate 0.30.1`）。

### 3.1 贯通验证（一步训练 + 一步验证 + 保存）

```bash
/opt/miniconda3/envs/vla/bin/python src/vla_finetune/train_lora.py \
  --config vla/configs/lora_finetune.yaml --smoke
```

`--smoke` 会把数据集裁到 `batch_size*2` 帧，执行一次前向 + 反向 + 优化器更新、一次验证前向、一次 adapter 保存，结果写入 `results/vla_finetune/smoke_result.json`。

### 3.2 完整训练

```bash
/opt/miniconda3/envs/vla/bin/python src/vla_finetune/train_lora.py \
  --config vla/configs/lora_finetune.yaml
```

命令行可覆盖 `--batch-size / --epochs / --max-steps / --out-dir / --device`；`--no-grad-ckpt` 可关闭梯度检查点（显存换速度）。训练结束保存 adapter + processor + 元信息到 `out_dir/checkpoint-<step>/`，汇总写入 `train_result.json`。

### 3.3 加载微调后的策略（推理衔接）

底座仍是 `openvla-7b`，只在加载后挂载 adapter：

```python
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor
import json, torch

meta = json.load(open("results/vla_finetune/checkpoint-<step>/finetune_meta.json"))
base = AutoModelForVision2Seq.from_pretrained(
    meta["model_dir"], trust_remote_code=True, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
base.norm_stats[meta["unnorm_key"]] = meta["norm_stats_entry"]  # 关键：注册本项目统计量
model = PeftModel.from_pretrained(base, "results/vla_finetune/checkpoint-<step>").to("cuda:0")
processor = AutoProcessor.from_pretrained("results/vla_finetune/checkpoint-<step>", trust_remote_code=True)

# 推理时 unnorm_key 必须用 meta["unnorm_key"]（pushcube_vla）
action7 = model.predict_action(**processor(...), unnorm_key=meta["unnorm_key"])
```

再用 `vla/utils/vla_adapter.py` 把 7 维动作映射回环境 4 维（丢弃旋转三维、位置 ×0.02、夹爪 `(g+1)/2`）。

## 4. 验证与输出

`--smoke` 已在容器 A6000 上真实跑通（模型 / 数据 / 训练代码全部无报错）。实测结果：

| 指标 | 实测值 |
|---|---|
| 模型加载 | 4.0 s，bf16，vocab_size=32000，n_action_bins=256 |
| 可训练参数 | 79,953,920 / 7,621,191,104（1.0491%） |
| 训练 loss（第 1 步） | 9.5334 |
| 梯度范数 / 学习率 | 13.0 / 5e-4 |
| 验证 loss / 动作 token 准确率 | 10.8420 / 0.1429（14 个 token） |
| 动作 token 编解码误差 | 0.00196（离散 bin 半宽内） |
| 显存保留 / 总耗时 | 15.465 GB / 24.05 s |
| adapter | `results/vla_finetune/checkpoint-0/`（已保存并写入 `finetune_meta.json`） |

完整机器可读结果见 `results/vla_finetune/smoke_result.json` 与 `results/vla_finetune/finetune_meta.json`。**注意**：这是单步贯通验证，loss 与准确率不构成收敛证据，骨架阶段只确认链路无报错，**不宣称任务成功率**。

## 5. 已知边界

1. 当前训练把 gripper 与旋转共 4 维常量 token 一起监督；若后续任务需要夹爪交互，应重建含抓取的数据集。
2. 训练用 `AutoProcessor` 直接处理 RGB 原图得到 6 通道 `pixel_values`；`PushCubeFrameDataset` 的 `image_dino/image_siglip` 双视图用于手工检查，二者归一化口径一致。
3. 数据集为脚本化专家轨迹，起点随机化 ±0.05 m；扩大场景多样性需重新采集，而非简单复制增强。
4. 全程不使用全量微调；显存不足优先降 batch + 增梯度累积，最后才考虑 QLoRA 并接受精度损失。
