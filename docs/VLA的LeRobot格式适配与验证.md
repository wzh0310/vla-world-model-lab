# LeRobot 格式适配与验证

本文档说明如何把预处理数据集转换为 LeRobot v0.7.0 标准格式（Parquet），并用官方接口加载验证。容器内的 lerobot 0.7.0 源码位于 `/opt/lerobot/src/lerobot`，字段约定均从源码实读核实。

## 字段规范与官方约定

`LeRobotDataset.create(repo_id, fps, features, root, ...)` 的 features 是「字段名 → dtype/shape」字典。核实到的关键约定（来自 `/opt/lerobot/src/lerobot` 源码）：

- `LeRobotDatasetMetadata.create` 会自动注入五个标准字段：`timestamp`、`frame_index`、`episode_index`、`index`、`task_index`，**不要在 features 中重复声明**。
- 图像字段：`{"dtype":"image","shape":(224,224,3),"names":[...]}`；本任务用 `use_videos=False`，图像内联存 Parquet（不编码 MP4）。
- 字符串字段：`{"dtype":"string","shape":(1,),...}`；`shape:()` 非法（实测报 `Corresponding feature is not valid`）。
- `add_frame(frame)` 需提供所有自定义字段 **加 `task` 键**（供官方生成 `task_index` 映射），且**不得**自行提供 `timestamp`/`frame_index`（由 writer 按 fps 自动算）。
- 每个 episode 调 `save_episode()` 落盘，全部完成后调 `finalize()` 写 footer，否则数据集无效。
- 字段名含 `.`（如 `observation.images.main`）是合法的；校验只禁止 `/`。

### 目标字段与来源

| 字段 | 来源 | 说明 |
|---|---|---|
| `episode_index` | 官方自动 | `save_episode` 顺序，每回合独立编号 |
| `frame_index` | 官方自动 | 每回合从 0 连续递增 |
| `timestamp` | 官方自动 | 每回合从 0 起，步长 1/fps（fps=50 → 0.02s） |
| `observation.images.main` | 预处理图像 | uint8 HWC → image feature，加载解码为 `[3,224,224]` |
| `observation.language_instruction` | 回合级指令 | string 字段，帧内共享，与 `task` 一致 |
| `action` | 归一化动作 | 4 维 `[dx,dy,dz,gripper]`，min-max 到 [-1,1] |
| `index` / `task_index` / `task` | 官方自动 | 全局帧索引 / 指令映射 / 指令文本 |

## 转换实现

`scripts/convert_to_lerobot.py`（lerobot 环境运行）：

1. 读预处理 `train.npz` / `val.npz` + 元信息；
2. 训练集与验证集分别建一个 `LeRobotDataset`（`root` 为 `lerobot/train`、`lerobot/val`），保证划分清晰、可独立加载；
3. 对每个 episode 按 `frame_index` 排序逐帧 `add_frame`，图像由 `[3,224,224]` float 还原为 uint8 HWC；
4. `save_episode()` 逐回合落盘，`finalize()` 收尾；
5. 写 `conversion_summary.json` 记录来源 episode_id 与规模。

数据集落在仓库外 `/data/wzh/robot/datasets/vla_pushcube_25/lerobot/`（不入库）。

## 验证实现

`scripts/verify_lerobot_dataset.py` 分两层校验（**42 项全部通过**）：

- **Parquet 列级**（快）：读 `data/chunk-*/file-*.parquet` 全部行，校验帧数、必需列、episode 数、每回合 `frame_index` 从 0 连续、`timestamp` 从 0 起且步长 1/50、`index` 全局连续、episode 按块排列、回合级指令帧内一致（每回合唯一指令）、指令非空、`task_index` 映射有效、动作维度与 [-1,1] 范围、训练/验证源 episode 无重叠。
- **官方接口抽样**（`LeRobotDataset[i]`）：校验加载帧数、`num_episodes`、字段齐全、图像解码形状 `[3,224,224]`、指令与 `task` 一致、抽样帧图像可稳定解码。

**重要约定**：LeRobot 的 `frame_index` 与 `timestamp` 是**每回合从 0 重排**，不是全局递增。实测 parquet 在回合边界处（`ep0 frame36 → ep1 frame0`）验证了这一点。若按「全局递增」校验会误报失败——这是校验假设错误，不是数据问题。

## 抽检证据

`results/vla_dataset/lerobot_sample_grid.png`：从转换后 parquet 抽 4 个 episode 首帧拼图，确认立方体/红色目标/机械臂/棋盘背景齐全、RGB 通道正常、指令标题清晰——图像在转换过程中保真、图文对齐。

## 产出与结果

```
lerobot/
├── train/                      # 20 回合 / 745 帧 / 16 个 task_index
│   ├── data/chunk-000/file-000.parquet
│   └── meta/{info,stats,tasks}.json|parquet + episodes/
├── val/                        # 5 回合 / 183 帧 / 5 个 task_index
├── conversion_summary.json
└── verify_result.json          # status: PASS, 42/42
```

训练集与验证集各自独立建库，`task_index` 按各自指令集合从 0 映射（train 16 个、val 5 个），这是 LeRobot 每库独立映射的正常行为，两库映射各自正确。

## 与后续环节的衔接

- 数据集已完全兼容 LeRobot 训练生态，可直接用 `LeRobotDataset(repo_id, root)` 加载，或接入官方训练脚本。
- `action` 为 min-max 归一化后的 4 维；微调时若 OpenVLA 需要 7 维，按旋转三轴补 0、夹爪二值化的口径扩维，并复用 `norm_stats.json` 做反归一化，勿混用 `unnorm_key` 的 q01/q99 统计量。
