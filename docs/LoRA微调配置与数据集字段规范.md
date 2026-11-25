# LoRA 微调配置调研与数据集字段规范

本文档调研 OpenVLA LoRA 微调的核心配置，并制定自定义 VLA 数据集的字段标准，为下一阶段构建域内数据与微调做准备。所有超参数与显存数据来自官方脚本、论文与社区实测（文末附来源），数据集规范对接本仓库现有轨迹结构与容器内的 LeRobot 0.7.0。

## 一、LoRA 微调核心配置

### 1.1 微调哪些模块

OpenVLA LoRA 微调**只挂 LLM 主干的线性层，冻结视觉编码器**：

- `target_modules = "all-linear"`：挂 Llama-2 主干的全部线性层（q/k/v/o、gate/up/down 投影）；官方实测比只挂 q/v 收敛更快，代价是可训练参数多一些。
- 视觉编码器（DINOv2 + SigLIP）保持冻结——它负责提取视觉特征，LoRA 只调整语言主干到动作解码的映射。
- 可训练参数约占全模型 **1.4%**，论文验证 LoRA 能匹敌全参数微调的性能。

### 1.2 推荐超参数（官方默认，经过 BridgeData V2 验证）

| 参数 | 官方默认 | 本项目建议 | 理由 |
|---|---|---|---|
| `lora_rank` | 32 | 32 | rank 对性能影响很小；16 无明显差异，64 只增显存 |
| `lora_alpha` | `min(rank,16)`（写死在源码） | 16 | 需改 rank 时才动；默认无需调整 |
| `learning_rate` | 5e-4（AdamW 恒定，无 warmup/衰减） | 5e-4 起步，可试 3e-4 + cosine | 1e-3 易震荡发散，1e-4 收敛慢一倍 |
| `batch_size` | 16（单卡理论上限） | 8 + 累积 2（48GB）/ 2 + 累积 8（24GB） | 等效 batch 16，显存减半；小 batch 不稳，必须配累积 |
| `epochs` / `max_steps` | 依数据量 | 小数据集 10000–30000 步 | 以 `action_accuracy` 与 `l1_loss` 收敛为准 |
| `image_aug` | True | True（小数据必开） | 关掉后 train loss 降到 0.4 但真机成功率反跌 |
| `shuffle_buffer_size` | 100000 | 小数据集 5000 | 缓冲拉满只白占内存 |

### 1.3 显存需求（关键：激活比权重更吃显存）

训练显存分四项，以 batch=8、序列长 288（256 图像 patch + prompt + 7 动作 token）实测为例：

| 显存项 | batch=8 用量 | 决定因素 | 压缩杠杆 |
|---|---|---|---|
| 底座权重 | 13.93 GiB | 参数量 × 每参数字节 | int8 / nf4 量化 |
| LoRA 权重+梯度+AdamW 状态 | 1.64 GiB | 可训练参数 × 16 字节 | rank 与 target_modules |
| **激活** | **18.34 GiB** | 层数 × 序列长 × batch × 隐藏维 | **梯度检查点** |
| logits + 交叉熵 | 0.69 GiB | batch × 序列长 × 词表 × 10 字节 | 只对动作位算 loss |

合计约 34 GiB。**激活（53%）大于权重（40%）**，这是最容易被忽略的一项。

**本项目硬件配置**：A6000 48GB 单卡，建议 `batch_size=8` + `grad_accumulation_steps=2` + 梯度检查点，等价 batch 16；不量化先跑通，显存不足再降级到 batch 4 + 累积 4（24GB 级别配置）。

### 1.4 QLoRA：训练量化与推理量化的差异（易混淆）

- **推理量化**：4-bit（nf4）实测匹配 bf16 的任务性能，是省显存部署的有效手段（本项目已验证 int4 权重 4.55 GB）。
- **训练量化（QLoRA）**：社区实测 `--use_quantization` 掉约 16 个点，官方注释标注 "CAUTION: 省显存但伤精度"。8-bit 量化在多数 GPU 上反而拖慢推理（量化算子开销），4-bit 因减少内存传输才划算。
- **结论**：A6000 48GB 上微调**不开量化**（bf16 权重 + LoRA 即可）；只有显存确实不足时才上 QLoRA，并接受精度损失。推理部署可用 int4。

### 1.5 训练/推理配置必须对齐

OpenVLA 系最隐蔽的一类 bug：模型加载成功、推理不报错、动作"看着像那么回事"，但成功率莫名低。根因通常是：

- `image_aug=True` 训练与推理的 center_crop 必须一致；
- `unnorm_key` 必须与微调数据集名一致（推理用错键则动作尺度全错）；
- 数据集构造时把图像 patch 位与 prompt 位 label 置 `-100`，**只对动作位算 loss**，定位预测位置靠 `labels != -100` 掩码，不能靠数 patch。

微调命令模板（官方 `vla-scripts/finetune.py`）：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name pushcube_vla \
  --run_root_dir ./runs \
  --adapter_tmp_dir ./adapter-tmp \
  --lora_rank 32 --batch_size 8 --grad_accumulation_steps 2 \
  --learning_rate 5e-4 --image_aug True --max_steps 20000 --save_steps 2000
```

## 二、微调数据集字段规范

### 2.1 现有数据盘点与缺口

本仓库 `outputs/data/trajectories.h5` 现有结构（每条轨迹一组）：

| 字段 | shape | dtype | 说明 |
|---|---|---|---|
| `images` | (T, 224, 224, 3) | uint8 | 主观相机 RGB 观测 |
| `actions` | (T, 4) | float32 | `[dx,dy,dz,gripper]`，量程 ±0.02m |
| `states` | (T, 13) | float32 | 关节/末端状态 |
| `rewards` | (T,) | float32 | 环境奖励 |
| `dones` | (T,) | bool | 回合结束标志 |

**核心缺口**：没有语言指令字段。VLA 微调必须有"图像 + 语言指令 → 动作"三元组，因此下阶段首要任务是给每条轨迹补 `language_instruction`。

### 2.2 目标数据格式（两套路径）

本项目涉及两条微调路线，数据格式要求不同，需同时满足：

**路径 A — OpenVLA LoRA**：官方要求 RLDS/TFDS 格式，需提供 `norm_stats`（每维 q01/q99）并注册 `unnorm_key`。动作必须是 7 维末端增量 `[dx,dy,dz,drx,dry,drz,gripper]`。

**路径 B — SmolVLA / LeRobot 生态**：使用 `LeRobotDataset` 格式（容器 `lerobot` 环境 v0.7.0，导入路径 `lerobot.datasets.lerobot_dataset`，`LeRobotDataset.create(repo_id, fps, features, ...)`）。这是后续选型的优先路线。

### 2.3 LeRobotDataset 字段标准（v0.7.0）

**必须字段**（框架自动注入）：

| 字段 | dtype | shape | 说明 |
|---|---|---|---|
| `timestamp` | float32 | (1,) | 回合内时间戳（秒） |
| `frame_index` | int64 | (1,) | 回合内帧序号 |
| `episode_index` | int64 | (1,) | 全局回合编号 |
| `index` | int64 | (1,) | 全局帧编号 |
| `task_index` | int64 | (1,) | 指向 `tasks.jsonl` 的任务索引 |

**本项目自定义字段**（基于 PushCube 现状定义）：

| 字段 | dtype | shape | names | 来源与说明 |
|---|---|---|---|---|
| `observation.image` | image/video | (224,224,3) | [height,width,channels] | 现有 `images[t]`，主观相机 |
| `observation.state` | float32 | (13,) | [state] | 现有 `states[t]` |
| `action` | float32 | (7,) | [action] | 现有 `actions[t]` 4 维扩到 7 维：位置 3 维 + 旋转 3 维补 0 + 夹爪 |
| `language_instruction` | string | (1,) | [instruction] | **新增**，每条轨迹一句英文指令 |

> 字段名不得含 `/` 分隔符（框架会报错）。动作扩维规则与 `vla/configs/pushcube_action.yaml` 及 `utils/vla_adapter.py` 的逆映射一致：PushCube 4 维 `[dx,dy,dz,gripper]` → OpenVLA 7 维 `[dx,dy,dz,0,0,0,gripper]`。

### 2.4 语言标注规范（核心新增项）

- **粒度**：每**条轨迹**一条指令，轨迹内所有帧共享同一指令（episode-level）。
- **语言**：英文（OpenVLA 预训练指令均为英文，中文会引入分布外语义）。
- **句式**：统一使用 `push/move ... to/onto ...` 的祈使句，与 VLA 模型预训练指令风格对齐。
- **指令集与 task_index 映射**（建议固定模板，便于泛化分析）：

| task_index | 指令模板 | 对应场景变化 |
|---|---|---|
| 0 | `push the cube to the red target` | 目标在 +x 方向 |
| 1 | `move the cube to the red circle` | 目标位置微调 |
| 2 | `push the block onto the red goal` | 目标措辞变化（同义泛化） |

- **避免的坑**：同一指令对应多种物理目标位置会让语义与任务解耦；应让指令措辞与目标位置一一对应，或明确记录为"同义泛化"对照组。

### 2.5 目录结构与元数据

LeRobotDataset 落盘结构：

```
pushcube_vla/
├── meta/
│   ├── info.json        # features/fps/codebase_version/shapes/names
│   ├── stats.json       # 各字段 mean/std/min/max（归一化用）
│   ├── tasks.jsonl      # task_index → 指令文本映射
│   └── episodes/        # 每回合元数据
├── data/                # 逐帧 parquet（state/action/timestamp）
└── videos/              # 相机画面 MP4（观察图像编码为视频）
```

**收集数量建议**：VLA LoRA 少量数据即可适配，但要有指令维度变化。建议每个 task_index 至少 50–100 条轨迹（合计 150–300 条），覆盖不同起点与光照（若可随机化）。本项目现有轨迹约 100+ 条但**无指令区分**，补标注时需按目标位置分配到对应 task_index。

### 2.6 数据集校验清单（下阶段执行依据）

1. 字段完整：五个必须字段 + 四个自定义字段全部存在；
2. 动作维度：7 维，旋转三维为 0，夹爪在 [0,1]；
3. 指令有效：`language_instruction` 非空、英文、在指令集内；
4. 统计一致：`stats.json` 的 action mean/std 与真实数据吻合；
5. 归一化口径：图像 [0,1] 或 [-1,1] 统一；动作按 `norm_stats` q01/q99；
6. train/test 划分：按 episode 划分（不能按帧，避免同轨迹泄漏）；
7. 可视化抽检：随机 10 帧核对图像-指令-动作对齐。

## 三、下一步执行计划

1. 写一个 HDF5 → LeRobotDataset 转换脚本，把现有 `images/states/actions` 映射为字段规范，并按目标位置分配 `language_instruction`；
2. 在 `lerobot` 环境（v0.7.0）用 `LeRobotDataset.create` 生成数据集并跑校验清单；
3. 选 SmolVLA / OpenVLA LoRA 其一跑微调（A6000 48GB：batch 8 + 累积 2 + 梯度检查点，不开量化）；
4. 微调后重跑 `vla/inference/rollout_vla_closed_loop.py`，对比成功率变化，验证域内数据是否解决零样本失败。

## 来源

- OpenVLA 官方微调脚本与模型卡（GitHub `openvla/openvla`、`scripts/finetune.py`，HuggingFace `openvla/openvla-7b`）
- OpenVLA 论文（LoRA 1.4% 参数匹敌全微调；4-bit 匹配 bf16、8-bit 反而变慢）
- 社区实测：古月居单卡 24G 微调 OpenVLA 显存账与 LoRA 默认参数、LoRA 微调全链路、4090 三模型实测
- LeRobot 数据集格式规范（Signalbotics / HuggingFace 课程 / LeRobot v0.7.0 `LeRobotDataset.create` 签名实测）
- 本项目容器实测：`lerobot==0.7.0` 导入路径与 `create` 签名；现有 HDF5 字段结构
