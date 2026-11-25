# PushCube VLA 数据集方案与指令模板库

本文档给出自定义 VLA 数据集的完整设计：标注规范依据、指令模板库、数据集规模与结构、字段定义、质量筛选标准以及与 LoRA 微调字段规范的衔接。配套模板库源码见 `vla/data/instruction_templates.py`，采集脚本见 `scripts/collect_vla_demos.py`。

## 一、标注规范依据（Open X-Embodiment）

参考 Open X-Embodiment（OXE，arXiv 2310.08864）与 RT-X 数据流水线，抽取四条对本项目直接适用的规范：

| 规范项 | OXE 做法 | 本项目落地 |
|---|---|---|
| 回合级指令 | `language_instruction` 是 episode 级字符串条件，与图像一起送入策略；任务完全靠文本串传达 | 每条轨迹一句英文指令，轨迹内所有帧共享 |
| 时序对齐 | 按固定控制频率逐帧采样，每帧一步动作，图像为动作执行前的观测 | 50 Hz，图像取 action 前观测，timestamp = 帧序/fps |
| 动作标准化 | 统一 7 维末端增量 `[dx,dy,dz,drx,dry,drz,gripper]`，先按数据集级统计量归一化再离散化 | 4 维 `[dx,dy,dz,gripper]` 扩为 7 维（旋转补 0），分位数统计存 `norm_stats` |
| 多样性与解耦 | 同一指令覆盖多样初始状态；同一初始状态可对应多样指令表述 | 起点 ±0.05 m 随机化；一条指令至少覆盖多条轨迹，5 类表述交叉分配 |

**关键边界**：OXE 只统一了存储 schema 与粗粒度 7 维动作接口，并未对齐各本体坐标系、相机外参与控制频率；因此统一格式不等于统一动力学。本项目只做算法层数据构建，动作含义沿用 PushCube 既有定义，不做底层控制映射。

## 二、核心原则

1. **同一指令 → 多样初始状态**：立方体起点在标称位置附近 ±0.05 m 均匀随机，机械臂初始位姿同步扰动，避免策略记住单一画面。
2. **同一状态 → 多样指令**：同一物理任务用 5 类措辞表达，指令维度与视觉/动作解耦，训练后模型对同义改写应给出一致动作。
3. **图文一致**：所有指令措辞（含反向参照类）的语义目标都是"把立方体送到红色目标区"，与脚本化专家轨迹的物理目标严格一致，不制造图文不符样本。
4. **质量优先于数量**：只保留成功到达且无穿模的轨迹，宁可少也不引入错误示范。

## 三、指令模板库（5 类 20 条）

来源：`vla/data/instruction_templates.py`。模板均只引用场景中真实存在的物体（蓝色立方体、桌面、红色圆形目标区），不引入分布外名词。

### 3.1 直白指令（direct）
| id | 指令 |
|---|---|
| D01 | push the cube to the red target area |
| D02 | push the cube onto the red target |
| D03 | move the cube to the red target disc |
| D04 | push the blue cube to the red circle |

### 3.2 简略指令（brief）
| id | 指令 |
|---|---|
| B01 | push block to goal |
| B02 | push cube to target |
| B03 | move block to red disc |
| B04 | push cube onto red goal |

### 3.3 描述性指令（descriptive）
| id | 指令 |
|---|---|
| S01 | move the blue cube on the desk to the circular mark |
| S02 | slide the blue cube across the table to the red circular target |
| S03 | push the small cube on the tabletop into the red marked area |
| S04 | move the blue block on the desk onto the red round marker |

### 3.4 变体指令（variant）
| id | 指令 |
|---|---|
| V01 | slide the cube forward to the red region |
| V02 | push the block ahead into the red target zone |
| V03 | shift the cube toward the red goal area |
| V04 | drive the cube to the red circular spot |

### 3.5 反向参照指令（reverse）
| id | 指令 |
|---|---|
| R01 | push the cube away from its starting position toward the red target |
| R02 | move the cube off its initial spot and into the red target area |
| R03 | push the block from where it starts to the red circular mark |
| R04 | slide the cube out of its starting position onto the red goal |

> 反向类用"远离起点 / 从初始位置移开"的反向参照措辞，但语义目标仍是红色目标区。这样既覆盖表述风格，又保证指令与示范动作一致。

### 3.6 指令分配策略

`sequence_for_count(25, seed=0)` 对 5 类轮询、类内轮询后确定性打散，得到 25 条指令：每类 4–6 条，20 条模板每条至少出现 1 次。分配结果落盘到 `instruction_ids` / `instructions` 字段，并写入 `tasks.jsonl`（template_id → 指令文本），便于按指令维度做泛化分析。

## 四、数据集规模与结构

### 4.1 规模

| 项 | 取值 | 依据 |
|---|---|---|
| 轨迹数 | 25 条成功轨迹 | 轻量化微调需求；同质小任务无需上百条 |
| 单条帧数 | 30–40 帧 | 接近/接近后推的脚本化轨迹实际长度 |
| 总帧数 | 约 800 帧（25×32 上下） | 满足 LoRA 少量数据适配 |
| 控制频率 | 50 Hz | 与 PushCube 环境 `timestep×frame_skip` 一致 |
| 图像 | 224×224×3 RGB，主观相机单视角 | 与 OpenVLA 输入分辨率一致 |
| 动作 | 4 维原始 + 7 维扩展 | 见字段定义 |

### 4.2 字段定义

原始数据以单个 `.npz` 落盘（仓库外，不入库）：

| 字段 | shape | dtype | 说明 |
|---|---|---|---|
| `images` | (N,224,224,3) | uint8 | 动作执行前的 RGB 观测 |
| `actions` | (N,4) | float32 | `[dx,dy,dz,gripper]`，位移±0.02 m |
| `states` | (N,13) | float32 | 关节/末端/立方体/目标状态 |
| `timestamps` | (N,) | float32 | **新增**，回合内时间戳（秒），每回合从 0 起 |
| `episode_lengths` | (25,) | int64 | 每条轨迹帧数（30–40） |
| `cube_starts` | (25,3) | float32 | 立方体起始位置（含随机偏移） |
| `arm_starts` | (25,3) | float32 | 末端起始位置（含随机偏移） |
| `episode_instructions` | (25,) | str | **新增**，每条轨迹一句英文指令 |
| `episode_template_ids` | (25,) | str | **新增**，模板 id（D01…R04） |
| `episode_success` | (25,) | bool | **新增**，是否到达目标 |
| `fps` | 标量 | int | 50 |

### 4.3 目录与划分

```
/data/wzh/robot/datasets/vla_pushcube_25/   # 仓库外，不入库
├── raw_demos_vla.npz          # 25 条对齐轨迹（上表全部字段）
├── instruction_manifest.json  # template_id/category/instruction 映射与分配
└── quality_report.json        # 每条轨迹 dist/steps/success/穿模检查
```

库内只保留脚本、模板库与文档（代码可复现），数据与权重一样不入库。train/test 按 episode 划分（不按帧），避免同一轨迹跨集泄漏；轻量化微调阶段可先全量训练，泛化评估时再按 20:5 划分。

## 五、质量筛选标准

生成候选轨迹后按以下顺序筛选，直到凑齐 25 条：

1. **到达目标**：终点立方体到目标中心距离 < 0.03 m；
2. **无穿模/瞬移**：相邻帧立方体平面位移 < 0.03 m（50 Hz 下约 1.5 m/s 上限），立方体 z 坐标保持在桌面高度合理区间，无穿入桌面或翻飞；
3. **帧数合规**：30 ≤ 帧数 ≤ 40；
4. **多样性**：起点偏移覆盖 ±0.05 m 的两个方向，指令 5 类均有覆盖。

未达标轨迹直接丢弃，不进入数据集；筛选结果写入 `quality_report.json` 供复现。

## 六、与 LoRA 微调规范的衔接

本方案产出是 `LoRA微调配置与数据集字段规范.md` 的数据准备前置步骤：

1. 本阶段 4 维动作在导出 LeRobotDataset 时扩为 7 维 `[dx,dy,dz,0,0,0,gripper]`，与 `pushcube_action.yaml` 逆映射一致；
2. `episode_instructions` 直接映射为 LeRobot 的 `language_instruction` / `tasks.jsonl` 的 `task_index`；
3. `timestamps` 对应 LeRobot 必须字段 `timestamp`，`episode_lengths` 对应 `episode_index`/`frame_index` 切分；
4. `norm_stats` 在格式化阶段按 7 维动作重新统计 q01/q99，注册 `unnorm_key` 供推理对齐。

## 七、复现命令

```bash
# 生成 25 条轨迹（mujoco 环境，EGL 离屏渲染）
$MUJOCO /data/wzh/robot/mujoco-studio-lab/scripts/collect_demos_v3.py gen \
    --episodes 120 --select 55   # 既有专家轨迹（v3）

# 本阶段专用采集（mujoco 环境）
$MUJOCO scripts/collect_vla_demos.py --episodes 120 --select 25 \
    --out /data/wzh/robot/datasets/vla_pushcube_25/raw_demos_vla.npz

# 指令分配/质量报告复核（vla 环境）
$VLA vla/data/instruction_templates.py
$VLA scripts/verify_vla_dataset.py --root /data/wzh/robot/datasets/vla_pushcube_25
```
