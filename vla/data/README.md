# PushCube VLA 数据集模块

本目录是 PushCube 视觉-语言-动作（VLA）数据集的**代码与接口模块**，配合仓库根目录 `scripts/` 下的采集、标注、预处理、格式转换脚本，构成从 MuJoCo 仿真轨迹到 OpenVLA / LeRobot 可消费数据的完整链路。

数据集本体（图像、动作、统计文件）体量较大，存放在仓库外 `/data/wzh/robot/datasets/vla_pushcube_25/`，不入库；本模块只沉淀**加载接口与字段规范**。

## 1. 数据集介绍

### 1.1 定位与目标

零样本 OpenVLA 在自建 PushCube 仿真场景成功率 0/3（域差距预期内，见 `docs/OpenVLA闭环推理测试与问题排查.md`）。本数据集提供**域内专家轨迹**，用于 LoRA 微调，让策略适配本场景的视觉分布与动作尺度。

设计原则：

- **轻量化优先**：20–30 条高质量差异化轨迹即可，不盲目扩大（边际收益低、训练时间高）。
- **指令多样性**：每个指令模板至少覆盖一条轨迹，同类句式做语义等价改写，提升语言泛化。
- **动作严格对齐**：归一化范围、增量语义、维度顺序与训练 / 推理 / 环境三者一致。

### 1.2 版本脉络与产物

| 阶段 | 产物 | 规模 | 落盘位置 |
|---|---|---|---|
| 原始采集 | `raw_demos_vla.npz` + manifest + quality | 25 轨迹 / 928 帧 | 仓库外数据集根 |
| 三元组标注 | `triplets/`（25 个独立 npz + 全局 npz + 索引表 + tasks.jsonl） | 25 轨迹 | `triplets/` |
| 预处理 | `preprocessed/train.npz` + `val.npz` + `norm_stats.json` + `splits.json` | 20 训练 / 5 验证（745 / 183 帧） | `preprocessed/` |
| LeRobot 适配 | `lerobot/{meta,data,...}`（v0.7.0 Parquet，官方接口可加载） | 20 训练 / 5 验证 | `lerobot/` |
| **v1.0 增强** | `v1.0/train_v1.npz` + `val_v1.npz` + `norm_stats.json` + `stats_report.json` | **80 训练块 / 2980 帧，5 验证 / 183 帧** | `v1.0/` |

当前推荐微调输入：`v1.0/train_v1.npz` + `v1.0/val_v1.npz` + `v1.0/norm_stats.json`。

### 1.3 采集与增强设置

- 专家控制律复用既有 `collect_demos_v3.py`；立方体起点 x/y 各 ±0.05 m、机械臂 xyz 扰动，终点距离全部 ≤ 0.028 m。
- 指令库：`instruction_templates.py` 定义 5 类 20 条模板（direct / brief / descriptive / variant / reverse），回合级指令（整条轨迹共享一句）。
- 图像增强（**仅训练集**）：亮度 U(0.85,1.15) + 对比度 U(0.85,1.15) + 高斯噪声 σ∈U(0,0.02)，每条轨迹参数固定以保证帧内一致。
- 指令增强：每条模板 2 条语义等价改写（与原指令共 3 种表述），均指向「把立方体送到红色目标区」同一物理任务。
- **验证集不做任何增强**，保持原始状态作为微调对比基线。

## 2. 字段说明

### 2.1 原始轨迹字段（`raw_demos_vla.npz`）

每个 episode 一组数组：

| 字段 | shape | dtype | 说明 |
|---|---|---|---|
| `images` | (T, 224, 224, 3) | uint8 | 全局相机 RGB 观测 |
| `actions` | (T, 4) | float32 | `[dx, dy, dz, gripper]`，位置量程 ±0.02 m |
| `states` | (T, 13) | float32 | 关节 / 末端状态 |
| `rewards` | (T,) | float32 | 环境奖励 |
| `dones` | (T,) | bool | 回合结束标志 |
| `timestamps` | (T,) | float64 | 帧时间戳（采集阶段新增） |
| `instruction` | (T,) | object(str) | 回合级英文指令（帧内一致） |
| `template_id` | (T,) | object(str) | 指令模板 id，如 `D01` |
| `success` | (T,) | bool | 终点达标标志 |

### 2.2 三元组字段（`triplets/episodes/epNNNN.npz`）

| 字段 | shape | dtype | 说明 |
|---|---|---|---|
| `instruction` | () | object(str) | 回合级指令（全帧共享） |
| `images` | (T, 224, 224, 3) | uint8 | 图像序列 |
| `actions` | (T, 4) | float32 | 动作序列 |
| `template_id` | () | object(str) | 模板 id |
| `task_index` | () | int64 | 模板 → task id 映射 |
| `episode_id` | () | int64 | 轨迹编号 |

索引表 `triplets_index.json` / `.csv` 记录每回合 `episode_id / instruction / length / action_dim / template_id`；`tasks.jsonl` 为 task_index → 指令文本映射。

### 2.3 预处理字段（`preprocessed/train.npz`、`v1.0/train_v1.npz`）

`PushCubeFrameDataset` 直接消费的逐帧数组：

| 字段 | shape | dtype | 取值 | 说明 |
|---|---|---|---|---|
| `images` | (N, 3, 224, 224) | float32 | [0,1] | RGB / CHW，bicubic resize |
| `actions` | (N, 4) | float32 | [-1,1] | `[dx,dy,dz,gripper]` 数据集级 min-max 归一化 |
| `instructions` | (N,) | object(str) | 英文 | 回合级指令 |
| `episode_ids` | (N,) | int64 | 0..N_ep-1 | 轨迹编号 |
| `frame_indices` | (N,) | int64 | 0..T-1 | 帧在轨迹内序号 |
| `task_indices` | (N,) | int64 | 0..19 | 模板 task id |
| `categories` | (N,) | object(str) | 5 类 | 指令类别 |
| `template_ids` | (N,) | object(str) | D01..R04 | 模板 id |

`norm_stats.json`（训练 / 验证共用一份）：

| 键 | 内容 | 用途 |
|---|---|---|
| `action.min` / `action.max` | 4 维物理量程 | 动作反归一化；`gripper` 为常量维（span=0） |
| `action.constant_dims` | `[3]` | 常量维标记 |
| `image.views.dino` | mean/std = ImageNet | DINOv2 视图归一化 |
| `image.views.siglip` | mean/std = 0.5 | SigLIP 视图归一化 |

> **关键**：`norm_stats.json` 是本项目自采数据的统计量，**不可与 OpenVLA 自带的 `unnorm_key`（如 `bridge_orig`）统计混用**。微调后推理须注册本项目 key（见 `src/vla_finetune/README.md`）。

### 2.4 LeRobot 字段（`lerobot/`，v0.7.0 Parquet）

| 字段 | dtype | shape | 来源 |
|---|---|---|---|
| `episode_index` | int64 | (1,) | 框架自动注入（全局回合号） |
| `frame_index` | int64 | (1,) | 框架自动注入（**每回合从 0 重排**） |
| `timestamp` | float32 | (1,) | 框架自动注入（步长 1/50 s） |
| `index` / `task_index` / `task` | — | — | 框架自动注入 |
| `observation.images.main` | image | (224,224,3) | 内联主视图（RGB） |
| `observation.language_instruction` | string | (1,) | 回合级指令 |
| `action` | float32 | (4,) | `[dx,dy,dz,gripper]` |

> LeRobot string feature 的 `shape` 必须为 `(1,)`，`shape:()` 会在 `create` 阶段报错。

### 2.5 动作空间对齐（4 维 ↔ 7 维）

环境接受 4 维增量 `[dx,dy,dz,gripper]`；OpenVLA 输出 7 维 `[dx,dy,dz,drx,dry,drz,gripper]`。扩维规则（训练 / 推理共用）：

```
4D [dx, dy, dz, g]  →  7D [dx, dy, dz, 0, 0, 0, g]
```

旋转三维补 0（推物无需旋转）；夹爪在环境中恒 0、在本数据集中保持归一化常量维。缩回 4 维时丢弃索引 3/4/5，位置按 `translation_scale_m=0.02` 缩放。完整映射见 `vla/configs/pushcube_action.yaml` 与 `vla/utils/vla_adapter.py`。

## 3. 生成方法

所有命令在容器 `wzh-ai` 内执行；`$R` = `/data/wzh/robot/vla-world-model-lab`，`$D` = `/data/wzh/robot/datasets/vla_pushcube_25`。环境变量隔离：`mujoco` 环境负责仿真采集，`vla` 环境负责标注 / 预处理 / 增强 / 训练，`lerobot` 环境负责格式验证。

### 步骤 1：采集（mujoco 环境）

```bash
MUJOCO_GL=egl /opt/miniconda3/envs/mujoco/bin/python scripts/collect_vla_demos.py \
  --episodes 180 --select 25 --seed 20261125 --out $D
/opt/miniconda3/envs/vla/bin/python scripts/verify_vla_dataset.py --root $D
/opt/miniconda3/envs/vla/bin/python scripts/visualize_vla_dataset.py
```

### 步骤 2：三元组标注（vla 环境）

```bash
/opt/miniconda3/envs/vla/bin/python scripts/build_vla_triplets.py
/opt/miniconda3/envs/vla/bin/python scripts/verify_vla_triplets.py --root $D/triplets
```

### 步骤 3：预处理与标准化（vla 环境）

```bash
/opt/miniconda3/envs/vla/bin/python scripts/preprocess_vla_data.py
/opt/miniconda3/envs/vla/bin/python scripts/verify_vla_preprocess.py --root $D/preprocessed
```

### 步骤 4：LeRobot 格式转换与验证（lerobot 环境）

```bash
/opt/miniconda3/envs/lerobot/bin/python scripts/convert_to_lerobot.py
/opt/miniconda3/envs/lerobot/bin/python scripts/verify_lerobot_dataset.py --root $D/lerobot
/opt/miniconda3/envs/lerobot/bin/python scripts/visualize_lerobot_sample.py
```

### 步骤 5：质量优化与增强 → v1.0（vla 环境）

```bash
/opt/miniconda3/envs/vla/bin/python scripts/audit_vla_quality.py
/opt/miniconda3/envs/vla/bin/python scripts/augment_vla_dataset.py
/opt/miniconda3/envs/vla/bin/python scripts/generate_vla_stats_report.py
/opt/miniconda3/envs/vla/bin/python scripts/verify_vla_v1.py
```

## 4. 使用方法

### 4.1 直接加载逐帧数据集

```python
import sys
sys.path.insert(0, "/data/wzh/robot/vla-world-model-lab")
from vla.data.pushcube_dataset import PushCubeFrameDataset

ds = PushCubeFrameDataset(
    npz_path="/data/wzh/robot/datasets/vla_pushcube_25/v1.0/train_v1.npz",
    norm_stats_path="/data/wzh/robot/datasets/vla_pushcube_25/v1.0/norm_stats.json",
    split="train",
)
item = ds[0]
# item: image [3,224,224], image_dino, image_siglip, action [4],
#       instruction, prompt, episode_id, frame_index, task_index, category, template_id
print(ds.unnormalize_actions(item["action"].numpy()))  # 还原物理增量
```

`__getitem__` 按 OpenVLA `preprocessor_config.json` 口径派生两个视觉视图（DINOv2 ImageNet 归一化、SigLIP 0.5 归一化）；微调脚本则用 `AutoProcessor` 直接处理 RGB 原图得到 6 通道融合 `pixel_values`，二者口径一致。

### 4.2 训练消费

LoRA 微调入口见 `src/vla_finetune/README.md`：

```bash
cd $R
/opt/miniconda3/envs/vla/bin/python src/vla_finetune/train_lora.py \
  --config vla/configs/lora_finetune.yaml --smoke
```

`--smoke` 仅跑一个训练步 + 一个验证步并保存 adapter，用于验证数据 / 模型 / 训练代码贯通；去掉该参数即按配置跑完整训练。

### 4.3 LeRobot 官方接口加载

```bash
/opt/miniconda3/envs/lerobot/bin/python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("pushcube_vla_train", root="/data/wzh/robot/datasets/vla_pushcube_25/lerobot/train")
print(len(ds), ds[0]["observation.language_instruction"])
PY
```

## 5. 统计信息（v1.0）

| 划分 | 轨迹数 | 帧数 | 单轨迹帧数 (min/mean/max) | 指令表述数 |
|---|---|---|---|---|
| 训练集 v1.0 | 20 源轨迹 → 80 增强块 | 2980 | 144 / 149.0 / 160 | 48 |
| 验证集（基线，不变） | 5 | 183 | 36 / 36.6 / 38 | 5 |

- **类别分布（训练）**：reverse 4 / direct 4 / descriptive 5 / variant 4 / brief 3；验证集 5 类各 1 条。
- **动作范围（归一化 [-1,1]）**：训练 dx/dy/dz 均达到 ±1，gripper 恒 0；验证 dy 下界 -0.9198、dz 下界 -0.9623。
- **图像统计**：训练像素 [0,1]，清晰度（Laplacian 方差）均值 796.03（最小 474.2）；验证均值 789.26（最小 568.71）。
- **质量抽查**：随机 3 轨迹（ep1/5/21）逐帧核对，指令一致、帧连续、物理跳变 ≤0.02 m（专家控制律量程内）、无模糊帧，全部 PASS；全局指令错位 0、越界 0、非有限值 0。
- **增强配置**：图像亮度/对比度 U[0.85,1.15]、噪声 σ∈U[0,0.02]，指令每条模板 2 条同义改写，seed=20261126。

完整报告（自动生成）：`results/vla_dataset/stats_report_v1.md`；机器可读版：`$D/v1.0/stats_report.json`。

## 6. 校验脚本一览

| 脚本 | 校验内容 | 结果 |
|---|---|---|
| `scripts/verify_vla_dataset.py` | 采集字段 / 达标 / 指令覆盖 | 全通过 |
| `scripts/verify_vla_triplets.py` | 三元组一一对应、回合级一致、索引一致 | 全通过 |
| `scripts/verify_vla_preprocess.py` | 双视图口径、反归一化误差、分层划分 | 28 项通过 |
| `scripts/verify_lerobot_dataset.py` | Parquet 列级时序 + 官方接口抽样解码 | 42 项通过 |
| `scripts/verify_vla_v1.py` | v1.0 规模 / 分布 / 动作范围 / 图像 / 质量 | 19 项通过 |

相关设计文档：`docs/VLA数据集方案与指令模板库.md`、`docs/VLA语言指令标注与三元组配对.md`、`docs/VLA数据预处理与标准化.md`、`docs/VLA的LeRobot格式适配与验证.md`、`docs/VLA数据集质量优化与增强.md`。
