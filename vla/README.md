# VLA 推理模块（v0.1）

基于 OpenVLA 的视觉-语言-动作推理模块，面向 MuJoCo PushCube 仿真场景。本模块把开源 VLA 模型接入自建仿真环境，完成从单帧推理到回合级闭环的全链路验证，并沉淀可复用的图像预处理、动作映射与模型封装工具。

## 版本与定位

- **模块版本**：v0.1（推理链路与工具模块首个正式版本）
- **模型**：OpenVLA 7B（`openvla/openvla-7b`），bf16 约 7.54B 参数
- **定位**：算法层与数据层验证；零样本闭环用于打通管线，不承担任务成功率指标
- **状态**：推理链路全部跑通；零样本三指令成功率 0/3（域差距预期内，详见测试结果）

## 目录结构

```
vla/
├── inference/               # 推理脚本（规范实现）
│   ├── verify_openvla_load.py        # 模型加载验证（bf16 / int4）
│   ├── infer_openvla_official_demo.py# 官方示例单帧推理 + 性能基线
│   ├── capture_pushcube_for_vla.py   # MuJoCo 截帧（mujoco 环境）
│   ├── infer_vla_action_for_mujoco.py# 单步推理与动作映射
│   ├── apply_vla_action_to_mujoco.py # 单步动作执行（mujoco 环境）
│   ├── rollout_vla_closed_loop.py    # 回合级闭环推理
│   └── plot_closed_loop.py           # 闭环结果可视化
├── utils/
│   └── vla_adapter.py        # 图像预处理 / 动作映射 / 结果解析 / 模型封装
├── data/
│   └── instruction_templates.py  # 5 类 20 条指令模板库与均衡分配
└── configs/
    ├── openvla.yaml          # 模型与推理参数、实测性能基线、依赖锁定
    └── pushcube_action.yaml  # 动作空间与映射口径
```

> 仓库根目录 `scripts/` 下同名脚本为兼容转发入口，内部调用 `vla/inference/` 的规范实现，旧命令保持可用。

## 环境依赖

容器内有多个隔离 conda 环境，本模块按职责分工：

| 环境 | 用途 | 关键依赖 |
|---|---|---|
| `vla` | 模型推理、数据处理、微调 | `transformers==4.40.1`、`tokenizers==0.19.1`、`timm==0.9.10`、`peft==0.11.1`、`accelerate==0.30.1` |
| `mujoco` | 仿真交互、截帧、演示数据 | `gymnasium==1.4.0`、`mujoco==4.0.1` |
| `lerobot` | 数据集格式验证（v0.7.0） | `lerobot==0.7.0` |

**依赖锁定理由**：OpenVLA 官方要求 `transformers==4.40.1`；`accelerate` 新版 1.x 会对 4bit 量化模型调用 `.to()` 报错，故锁 0.30.1。`pip check` 实测通过。

**模型权重**（仓库外，不入库）：`/data/wzh/robot/models/openvla-7b`，15.09 GB / 18 文件，来自 ModelScope 镜像 `zixiaoBios/openvla-7b`（官方源在国内容器仅 0.03 MB/s）。下载方式见 `docs/OpenVLA环境与模型准备说明.md`。

## 快速上手

以下均在容器 `wzh-ai` 内执行，`$R` 代表仓库根 `/data/wzh/robot/vla-world-model-lab`。

### 1. 验证模型加载

```bash
cd $R
/opt/miniconda3/envs/vla/bin/python vla/inference/verify_openvla_load.py \
  --model-dir /data/wzh/robot/models/openvla-7b
# 低显存卡追加 --quant int4
```

### 2. 官方示例单帧推理

```bash
/opt/miniconda3/envs/vla/bin/python vla/inference/infer_openvla_official_demo.py \
  --repeats 5
```

### 3. 自定义场景单步闭环（三段式，跨环境）

```bash
# mujoco 环境：截帧并保存可复现初始状态
/opt/miniconda3/envs/mujoco/bin/python vla/inference/capture_pushcube_for_vla.py
# vla 环境：推理并映射动作
/opt/miniconda3/envs/vla/bin/python vla/inference/infer_vla_action_for_mujoco.py \
  --instruction "push the cube to the red target"
# mujoco 环境：恢复状态执行一步
/opt/miniconda3/envs/mujoco/bin/python vla/inference/apply_vla_action_to_mujoco.py
```

### 4. 回合级闭环推理（单进程，推荐）

```bash
/opt/miniconda3/envs/vla/bin/python vla/inference/rollout_vla_closed_loop.py \
  --max-steps 30 --out-dir results/vla_closed_loop --save-frames --diagnose
/opt/miniconda3/envs/vla/bin/python vla/inference/plot_closed_loop.py
```

闭环采用**单进程注入**方案：`vla` 进程内注入 `mujoco` 环境 site-packages，模型只加载一次、环境全程驻留。两环境 `torch==2.6.0+cu124` / `numpy==2.5.5` 版本一致，无 ABI 冲突。

### 5. 用兼容入口（旧命令）

```bash
/opt/miniconda3/envs/vla/bin/python scripts/rollout_vla_closed_loop.py --help
```

## 动作映射说明

OpenVLA 输出 7 维末端增量，PushCube 只接受 4 维动作，映射分四步（`utils/vla_adapter.py::map_action`）：

| 步骤 | 操作 | 说明 |
|---|---|---|
| ① 还原归一化 | `unit = clip(2*(raw-q01)/(q99-q01) - 1, -1, 1)` | 用 `bridge_orig` 数据集 q01/q99；`unnorm_key` 选错则尺度全错 |
| ② 丢弃旋转 | 去掉 `drx/dry/drz` 三维 | 推物任务无需旋转 |
| ③ 位置缩放 | `env_xyz = unit[:3] × 0.02` | 每步末端位移上限 ±0.02 m，与环境 `ACTION_LOW/HIGH` 对齐 |
| ④ 夹爪映射 | `gripper = (g+1)/2` | 从 [-1,1] 到 [0,1]；本任务接口保留但不参与动力学 |

**诊断开关**：`--diagnose` 会在首步做两组对照——RGB vs BGR（通道顺序）、正常 vs 翻转 xyz（方向反转），结果写入每回合 `episode.json` 的 `diagnosis` 字段。实测通道错误使 y 分量偏移 0.028 m，方向翻转差异 0.039 m，证明两个环节都必须严格正确。

## 测试结果

三条指令各 30 步（A6000，bf16，相同初始状态，立方体起点 `[0.62,-0.12]`、目标 `[0.82,-0.12]`）：

| 指令 | 距离变化 | 总奖励 | 平均单步耗时 |
|---|---|---|---|
| `push the cube to the red target` | 0.200 → 0.200 | -1.66 | 0.305 s |
| `move the cube to the red circle` | 0.200 → 0.200 | -1.63 | 0.301 s |
| `push the block onto the red goal` | 0.200 → 0.200 | -1.82 | 0.287 s |

**成功率 0/3**。关键判读：

- **管线通畅**：每步 EE 实动与映射动作一致，30 步累计位移 0.14–0.24 m；
- **语言条件生效**：三条指令动作方向不同——`red target/circle` 指向 +x（正确朝目标方向），`red goal` 明显不同；
- **为何没成功**：预训练于真实机器人数据，仿真域差距大；零样本动作缺长程规划，末端未接触立方体。这是域差距预期内结果，非管线 bug。

**性能基线**：bf16 单步 0.305 s（3.65 Hz）、显存 15.13 GB 分配 / 15.53 GB 保留；int4 显存 4.55 GB 但单步 1.31 s。A6000 上优先 bf16。

完整数据与图表见 `results/vla_closed_loop/` 与 `docs/OpenVLA闭环推理测试与问题排查.md`。

## 自定义 VLA 数据集（微调前置）

为零样本闭环失败提供域内数据，已构建一套轻量化专家数据集：

| 项 | 值 |
|---|---|
| 轨迹 / 帧数 | 25 条 / 928 帧（单条 36–40 帧，均值 37.1） |
| 指令 | 5 类共 20 条模板，每条模板至少 1 条轨迹（全覆盖） |
| 起点随机化 | 立方体 x/y ±0.05 m、机械臂 xyz 扰动；终点 dist ≤ 0.028 m 全部达标 |
| 字段 | 图像 / 4 维动作 / 13 维状态 / **时间戳** / 指令文本 / 模板 id / 达标标志 |
| 落盘 | 仓库外 `/data/wzh/robot/datasets/vla_pushcube_25/`（npz + manifest + quality + verify，不入库） |

```bash
MUJOCO_GL=egl /opt/miniconda3/envs/mujoco/bin/python scripts/collect_vla_demos.py \
  --episodes 180 --select 25 --seed 20261125 --out /data/wzh/robot/datasets/vla_pushcube_25
/opt/miniconda3/envs/vla/bin/python scripts/verify_vla_dataset.py --root /data/wzh/robot/datasets/vla_pushcube_25
/opt/miniconda3/envs/vla/bin/python scripts/visualize_vla_dataset.py
```

采集复用既有 `collect_demos_v3.py` 的专家控制律，新增起点大幅随机化、时间戳、指令分配与穿模筛选。
完整设计与字段规范见 `docs/VLA数据集方案与指令模板库.md` 与 `docs/LoRA微调配置与数据集字段规范.md`。

## 已知限制与下一步

1. 零样本无法完成任务，需用上述域内数据做 LoRA 微调（配置见 `docs/LoRA微调配置与数据集字段规范.md`）；
2. 推理速度瓶颈在自回归生成 7 个动作 token（约 3.4 Hz），换流匹配模型（π0 / SmolVLA）可提升吞吐；
3. 单步闭环使用跨本体启发式映射，非物理标定；微调阶段应改用真实动作空间对齐；
4. 数据集为脚本化专家轨迹、夹爪恒为 0（推物任务无需抓取）；扩维到 7 维时旋转补 0、夹爪二值化。
