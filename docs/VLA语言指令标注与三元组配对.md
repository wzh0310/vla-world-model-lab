# VLA 语言指令标注与三元组配对

本文档说明回合级指令机制、三元组数据构建与数据索引表设计。配套脚本：`scripts/build_vla_triplets.py`（构建）、`scripts/verify_vla_triplets.py`（校验），模板库见 `vla/data/instruction_templates.py`。

## 一、回合级指令机制（核心原则）

VLA 采用**单条指令对应整段轨迹**：一条演示轨迹从头到尾共享同一句语言指令，轨迹内每一帧都条件于这条指令，而不是逐帧更换指令。

| 维度 | 回合级（本项目采用） | 逐帧级（不采用） |
|---|---|---|
| 指令粒度 | 一条轨迹一句指令 | 每帧可不同指令 |
| 语义 | 描述整个任务目标（"把立方体推到红区"） | 会把同一任务切成多段矛盾语义 |
| 与 OXE 对齐 | RLDS 的 `language_instruction` 是 episode/step 恒定字符串 | 与主流 VLA 数据集不一致 |
| 工程影响 | 一条指令绑定整段动作序列，易于配对与检索 | 标注成本高、语义噪声大 |

**例外说明**：长程多阶段任务（如"先开门再取物"）在部分数据集会用阶段级指令，但 PushCube 是单阶段原子任务，回合级是正确选择。

在中间格式里，指令以两层视图保存：
- `episode_instructions`（长度 = 轨迹数）：回合级，每轨迹一句；
- `frame_instructions`（长度 = 总帧数）：按 `frame_episode_ids` 展开的帧级视图，供逐帧采样器直接取用。

两者由同一源生成、语义严格一致，校验脚本逐轨迹验证帧级指令完全相同。

## 二、三元组构建

### 2.1 三元组定义

每条轨迹是一个三元组 **（指令文本，图像序列，动作序列）**：

```
episode i = (
  instruction = "push the cube onto the red target",   # 回合级，1 句
  images      = (T_i, 224, 224, 3) uint8,              # 图像序列，T_i 帧
  actions     = (T_i, 4) float32,                      # 动作序列 [dx,dy,dz,gripper]
)
```

附带 `states (T_i,13)`、`timestamps (T_i,)`、模板 id、类别、task_index 作为元数据。

### 2.2 指令分配

采集阶段已用 `instruction_templates.sequence_for_count()` 做确定性均衡随机分配（25 条覆盖全部 20 条模板、5 类齐全）。构建脚本提供两种模式：

- `reuse`（默认）：沿用预分配，保持数据与既有 `instruction_manifest.json` 一致；
- `reshuffle --seed N`：按 seed 重新随机分配，用于演示"给每条轨迹随机分配 1 条模板"的机制。

两种模式都保证：每条轨迹恰好一条指令、指令与轨迹一一对应、指令来自模板库。

### 2.3 中间格式（仓库外，不入库）

```
/data/wzh/robot/datasets/vla_pushcube_25/triplets/
├── triplets.npz           # 全局打包：图像/动作/状态/时间戳 + 回合级与帧级指令
├── triplets_index.json    # 数据索引表（JSON）
├── triplets_index.csv     # 数据索引表（CSV）
├── tasks.jsonl            # task_index -> 指令文本（对接 LeRobot）
├── verify_result.json     # 校验结果
└── episodes/
    ├── ep0000.npz         # 每条轨迹独立三元组（指令与轨迹一一对应）
    └── ...                # ep0001 ~ ep0024
```

### 2.4 数据索引表

索引表逐条记录，字段如下：

| 字段 | 说明 |
|---|---|
| `episode_id` | 全局轨迹编号 0–24 |
| `instruction` | 该轨迹的语言指令（回合级） |
| `template_id` | 模板 id（D01…R04） |
| `category` | 表述类别（direct/brief/descriptive/variant/reverse） |
| `task_index` | 模板稳定编号，对接 LeRobot `tasks.jsonl` |
| `traj_length` | 轨迹帧数（36–40） |
| `action_dim` | 动作维度（4） |
| `state_dim` | 状态维度（13） |
| `image_shape` | 图像形状 [224,224,3] |
| `fps` | 控制频率（50） |
| `frame_start` / `frame_end` | 该轨迹在全局数组中的帧区间 |
| `final_dist` | 终点到目标距离（均 < 0.03） |
| `file` | 独立三元组文件路径 |

`task_index` 用模板 id 稳定编号（同一指令共用一个索引），使 LeRobot 格式化时 `tasks.jsonl` 与指令一一对应，避免同义指令被拆成多个任务。

## 三、校验

`scripts/verify_vla_triplets.py` 检查六项：字段与形状对齐、指令-轨迹一一对应（模板库内、类别匹配）、回合级帧内一致、25 个独立文件与全局切片一致、索引表 JSON/CSV 一致、tasks.jsonl 映射正确。全部通过后写 `verify_result.json`（`status: PASS`）。

```bash
$VLA scripts/build_vla_triplets.py --raw /data/wzh/robot/datasets/vla_pushcube_25/raw_demos_vla.npz \
  --out /data/wzh/robot/datasets/vla_pushcube_25/triplets
$VLA scripts/verify_vla_triplets.py --root /data/wzh/robot/datasets/vla_pushcube_25/triplets
```

## 四、与后续阶段的衔接

三元组中间数据集是 LeRobot 格式化的直接输入：

1. `instruction` → LeRobot `language_instruction` / `tasks.jsonl` 的 `task_index`；
2. `images` → `observation.images.main`（视频编码）；
3. `actions`（4 维）→ 扩维为 7 维 `[dx,dy,dz,0,0,0,gripper]`；
4. `timestamps` → LeRobot 必须字段 `timestamp`；
5. `traj_length` / `episode_id` → `frame_index` / `episode_index` 切分依据。

下一步在 `lerobot` 环境（v0.7.0）用 `LeRobotDataset.create` 完成格式化并跑字段校验清单。
