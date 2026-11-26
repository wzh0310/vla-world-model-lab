# VLA 指令与场景泛化数据集模块

本模块产出用于 OpenVLA LoRA 微调的标准化数据集，承接既有 25 条 PushCube 基础轨迹，
新增「推到目标区偏左 / 偏右」两类空间变体，并把每条物理轨迹展开为多条语言条件化
轨迹，最终以 HuggingFace Dataset 格式落盘，供微调骨架直接加载。

所有真实数据落在仓库外 `/data/wzh/robot/datasets/vla_pushcube_finetune/`，不入库；
本目录只放可复现的构建 / 转换 / 校验脚本。

## 目录

| 文件 | 运行环境 | 作用 |
| --- | --- | --- |
| `instruction_variants.py` | 任意 | 左右变体的目标坐标、类别、每条 4 条同义指令 |
| `collect_variants.py` | `mujoco` | 用 `PushCubeEnv(scene_override=True)` 注入目标坐标采集变体轨迹 |
| `build_hf_dataset.py` | `lerobot` | 指令展开 + 动作归一化 + 轨迹级 8:1:1 划分 + 转 HF Dataset |
| `verify_hf_dataset.py` | `lerobot` | 校验字段 / 图像 / 动作 / 类别 / 划分无泄漏 |
| `hf_dataset_adapter.py` | `vla` | 把 HF Dataset 包装为 PyTorch Dataset，直接对接 `FinetuneCollator` |
| `verify_adapter.py` | `vla` | 校验适配器 + collator 产出合法 batch（不加载 7B 模型） |

## 字段说明

HuggingFace Dataset 三个 split（`train` / `val` / `test`）字段统一为：

| 字段 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- |
| `image` | `Image` (RGB) | 224×224×3 | 与 OpenVLA 输入分辨率对齐的顶视相机帧 |
| `instruction` | `string` | 标量 | 回合级自然语言指令（每条轨迹的每条同义表述生成一个样本） |
| `action` | `float32` | (4,) | 归一化增量动作 `[dx, dy, dz, gripper]`，范围 [-1,1] |

动作归一化沿用数据集级 min-max：`x_norm = 2*(x-min)/(max-min) - 1`，
常量维（gripper 恒为 0）置 0。参数写在 `hf_dataset/norm_stats.json`，
微调与推理必须复用同一份，避免与 OpenVLA 自带的 `unnorm_key` 统计量混用。

4 维动作对接 OpenVLA 7 维动作空间时按 `[dx, dy, dz, 0, 0, 0, gripper]` 扩维，
与既有 LoRA 微调骨架 `src/vla_finetune/` 的口径完全一致。

## 生成流程

按环境分工依次执行（三步不可混用环境）：

```bash
# 1) mujoco 环境：采集左右目标变体（EGL 离屏渲染）
MUJOCO_GL=egl /opt/miniconda3/envs/mujoco/bin/python \
  vla_finetune/data/collect_variants.py \
  --episodes 160 --select 10 --seed 20261126 \
  --out /data/wzh/robot/datasets/vla_pushcube_finetune

# 2) 指令展开 + 归一化 + 8:1:1 划分 + 转 HF Dataset
#    用 vla 环境生成：微调在 vla 环境消费，须让落盘 datasets 版本与消费端一致
#    （lerobot 环境 datasets 5.2 会写出 List 类型，vla 环境 3.6 无法加载；
#     vla 3.6 写出 Sequence，两端语义一致且微调端可原生读取）。
/opt/miniconda3/envs/vla/bin/python \
  vla_finetune/data/build_hf_dataset.py \
  --base /data/wzh/robot/datasets/vla_pushcube_25/raw_demos_vla.npz \
  --variants /data/wzh/robot/datasets/vla_pushcube_finetune/variants \
  --out /data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset

# 3) 校验数据集 + 适配器（vla 环境）
/opt/miniconda3/envs/vla/bin/python \
  vla_finetune/data/verify_hf_dataset.py \
  --data /data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset
/opt/miniconda3/envs/vla/bin/python \
  vla_finetune/data/verify_adapter.py \
  --data /data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset \
  --repo /data/wzh/robot/vla-world-model-lab
```

## 划分策略

按「物理轨迹」分组做分层 8:1:1：同一条物理轨迹的所有同义指令表述整体落在同一
split，杜绝同轨迹跨集泄漏；按类别（`direct`/`brief`/`descriptive`/`variant`/
`reverse`/`variant_left`/`variant_right`）分层，保证各 split 指令分布一致，随机种子
固定可复现。

## 加载方法

```python
from datasets import load_from_disk
ds = load_from_disk("/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset/train")
row = ds[0]
img = row["image"]              # PIL.Image RGB 224x224
instruction = row["instruction"]  # str
action = row["action"]          # list[float] length 4, in [-1,1]
```

统计信息与校验结论见 `results/` 下的本阶段报告；数据集本身不入 git。
