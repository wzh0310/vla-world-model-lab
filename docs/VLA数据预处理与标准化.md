# VLA 数据预处理与标准化

本文档说明如何把「指令 - 图像序列 - 动作序列」三元组中间数据集，处理为完全匹配 OpenVLA 输入输出的格式，并产出可被 PyTorch DataLoader 直接加载的数据集与归一化参数文件。

## 目标与口径来源

预处理不自行发明口径，而是从模型侧配置文件实读：`/data/wzh/robot/models/openvla-7b/preprocessor_config.json`。该文件表明 OpenVLA 使用**双视觉主干**（fused vision backbone），同一张图送进两个视图：

| 视图 | 主干 | mean | std |
|---|---|---|---|
| dino | DINOv2-ViT-L/14 | `[0.485, 0.456, 0.406]` | `[0.229, 0.224, 0.225]` |
| siglip | SigLIP-ViT-So400M/14 | `[0.5, 0.5, 0.5]` | `[0.5, 0.5, 0.5]` |

两路输入尺寸均为 `3×224×224`，resize 策略 `resize-naive`（bicubic）。因此本预处理把像素统一到 `[0,1]` 后，在 Dataset 内按上表派生 `image_dino` / `image_siglip` 两个张量，与推理端 `processor(prompt, image)` 的行为严格一致。

## 图像预处理

1. 输入为三元组中的 RGB 观测（uint8，HWC）；
2. 显式 `resize(224,224)`（bicubic）——源数据虽已是 224，仍显式执行以锁定口径，避免上游尺寸变化导致静默错位；
3. 通道转为 CHW，像素除以 255 归一化到 `[0,1]`，dtype float32；
4. 双视图 mean/std 与口径写入 `norm_stats.json`，Dataset 读取时据此派生，不硬编码在训练代码里。

## 动作预处理

- 统一为 4 维增量动作 `[dx, dy, dz, gripper]`；旋转三轴在本任务不使用，已在采集阶段丢弃。
- 数据集级 min-max 归一化到 `[-1,1]`：

  ```
  x_norm = 2 * (x - min) / (max - min) - 1
  x      = min + (x_norm + 1) * (max - min) / 2
  ```

- `min` / `max` / `span` 在**全部 25 条轨迹（928 帧）**上统计一次，训练集与验证集共用同一份参数，避免分别统计造成分布泄漏。
- `gripper` 维在本脚本化推物任务中恒为 0，属常量维（`span < 1e-8`）。归一化时置 0，反归一化时还原常量，避免除零。`constant_dims` 已写入参数文件。
- `norm_stats.json` 是**微调与推理共用的唯一真源**：微调时用它把归一化动作还原成物理量；推理端再按 OpenVLA 的 `unnorm_key` 口径映射到本环境量程。注意 OpenVLA 原生活化用的是数据集 q01/q99 统计量，而本轻量数据集采用 min-max——两者都是把网络输出映射回动作区间的手段，不可混用同一数值，须通过各自的参数文件转换。

## 训练 / 验证划分

采用**按指令类别分层**的划分：5 类各抽 1 条组成 5 条验证集，其余 20 条为训练集。这样验证集完整覆盖 5 类表述、训练集保持与整体相近的类别比例，避免随机划分造成某类在验证集缺席。类别内抽取用固定 `seed=42`，结果可复现，`splits.json` 记录 episode_id 列表与类别分布。

## PyTorch Dataset / DataLoader

`vla/data/pushcube_dataset.py` 提供 `PushCubeFrameDataset`，按索引读取单帧三元组，返回：

- `image` / `image_dino` / `image_siglip`：`[3,224,224]` 张量；
- `action`：`[4]` 归一化动作；
- `instruction`：回合级指令（同一轨迹所有帧共享）；
- `prompt`：拼好的完整 OpenVLA 输入文本；
- `episode_id` / `frame_index` / `task_index` / `category` / `template_id`：追踪元信息。

```python
from torch.utils.data import DataLoader
from vla.data.pushcube_dataset import PushCubeFrameDataset

ds = PushCubeFrameDataset("preprocessed/train.npz", "preprocessed/norm_stats.json")
dl = DataLoader(ds, batch_size=16, shuffle=True)
batch = next(iter(dl))
# batch["image_dino"]  -> [16,3,224,224]
# batch["image_siglip"]-> [16,3,224,224]
# batch["action"]      -> [16,4]
```

## 产出与校验

```
preprocessed/
├── train.npz            # 20 条轨迹展开的帧级样本
├── val.npz              # 5 条轨迹展开的帧级样本
├── norm_stats.json      # 动作 min/max/span、常量维、图像双视图参数
├── splits.json          # 划分策略、seed、episode_id、类别分布
└── verify_result.json   # 校验结果
```

`scripts/verify_vla_preprocess.py` 覆盖：字段完整、图像/动作形状与范围、20/5 划分无重叠、类别分层覆盖、回合级帧内一致、归一化可逆（反归一化误差 < 1e-6）、常量维标记、Dataset 单帧读取、DataLoader 批加载。全部通过后才能进入后续格式化与微调。

## 与后续环节的衔接

- 数据集可直接进入 LeRobot 格式适配：`prompt`/`instruction` → `language_instruction`，4 维动作扩为 7 维时旋转三轴补 0、gripper 按二值化口径处理，时间戳取自三元组原始字段。
- 进入 LoRA 微调前，务必确认训练代码读取的是本目录 `norm_stats.json`，而非模型的 `unnorm_key` 统计量；两者口径不同，混用会导致动作尺度错误。
