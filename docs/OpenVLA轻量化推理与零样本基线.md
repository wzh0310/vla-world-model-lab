# OpenVLA 轻量化推理与零样本基线

本文档记录轻量化依赖核验、OpenVLA 4bit 单样本推理、以及在自定义 PushCube 场景上的零样本基线。所有数值来自容器 `wzh-ai` 内 A6000 的真实运行，未使用估计值；零样本成功率只作为微调前的对照基线，**不作为任务成功或收敛结论**。

## 1. 环境依赖核验

在 `vla` 环境直接核验手册要求的四个核心依赖（均已安装，无需重复 pip install，避免破坏已锁定的版本组合）：

| 包 | 版本 |
| --- | --- |
| peft | 0.11.1 |
| bitsandbytes | 0.50.2 |
| datasets | 3.6.0 |
| accelerate | 0.30.1 |
| torch | 2.6.0+cu124（CUDA 可用） |
| transformers | 4.40.1 |

说明：`accelerate` 必须保持在 0.30.x，升级到 1.x 会对 4bit 模型调用 `.to()` 并报错；此约束已写入 `vla/configs/openvla.yaml`。

## 2. 模型权重与量化加载

- 权重目录：`/data/wzh/robot/models/openvla-7b`（仓库外，不入库）。
- 实际来源：ModelScope 镜像 `zixiaoBios/openvla-7b`，与手册建议的 `openvla/openvla-7b-prismatic` 为同一 7B prismatic 权重；选用本地镜像是为了规避 HuggingFace 下载不稳定，权重内容一致。
- 配置核对：`model_type=openvla`、`architectures=OpenVLAForActionPrediction`、`norm_stats` 含 `bridge_orig` 等 25 个键。
- 量化配置：NF4 4bit + 双量化，计算 dtype 为 bf16；显式 `device_map`，量化后不调用 `.to()`；视觉塔输入用 fp16。

实测资源占用：

| 指标 | bf16（既有基线） | int4（本次） |
| --- | --- | --- |
| 参数量（加载后统计） | 7.54 B | 3.90 B（4bit 存储口径） |
| 单样本显存保留 | 15.532 GB | **4.823 GB** |
| 闭环单步平均耗时 | 0.2976 s | 0.4351 s |

结论：int4 满足单卡显存 ≤8 GB 的要求，代价是单步耗时约增加到 1.5 倍。

## 3. 单样本推理脚本

新增 `vla/inference/inference.py`，输入输出链路：

1. 输入：单张 224×224 RGB 图像（文件或实时环境观测）+ 自然语言指令；
2. 复用 `vla/utils/vla_adapter.py` 的 `OpenVLAPolicy` 完成 prompt 构造、推理与动作映射；
3. 输出：OpenVLA 原始 7 维动作 → 反归一化 → PushCube 4 维物理动作（丢弃 3 维旋转）；
4. 校验：动作幅度是否落在环境动作空间内、水平方向与目标方向的夹角余弦；
5. 在 MuJoCo 中执行一步，记录末端位移、立方体位移与目标距离变化。

单样本实测（`--quant int4`，指令 `push the cube to the red target`，图像与初始状态来自既有单步产物）：

- 原始 7 维：`[0.00359, 0.00249, -0.00544, -0.064, 0.04544, -0.0452, 0.0]`
- 映射 4 维：`[0.00267, 0.00141, -0.00753, 0.0]`
- 幅度合法：是（最大单轴位移 0.00753 m ≤ 0.02 m，夹爪 0 ∈ [0,1]）
- 方向余弦：**0.8838**（指向目标）
- 环境一步：末端位移 `[0.012, 0.001412, -0.007529]`，立方体位移 `[0, 0]`，目标距离 0.2 → 0.2
- 推理耗时 0.5464 s，显存保留 4.823 GB

结果文件：`results/vla_inference_single/inference_result.json`。动作幅度与方向合理，但单步未建立与立方体的接触，因此立方体未移动——这与闭环基线现象一致。

## 4. 零样本基线（修正后）

### 4.1 关键问题：离屏渲染未初始化导致旧基线无效

核对发现：既有 `results/vla_closed_loop/summary.json`（bf16，成功率 0）运行时未设置 `MUJOCO_GL=egl`，环境返回 `obs["image"]` 为 `shape=(1,)` 的全零占位，即模型实际看到的是空白图像。因此该结果**不能作为有效的视觉零样本基线**。

修复措施：

- 在新 `inference.py` 的实时取图路径加入显式拦截：图像 shape 不为 `(224,224,3)` 或全零时抛出错误，并提示设置 `MUJOCO_GL=egl`；
- 重新以 `MUJOCO_GL=egl` 运行闭环，确认观测为真实图像（`shape=(224,224,3)`，像素范围 33–255，均值 141.475）。

### 4.2 有效基线结果

配置：int4、`unnorm_key=bridge_orig`、量程 ±0.02 m、每回合 30 步、3 条同义指令、种子 0/1/2。

| 指令 | 距离变化 | 成功 | 总奖励 | 单步耗时 |
| --- | --- | --- | --- | --- |
| push the cube to the red target | 0.2000 → 0.2000 | 否 | -1.759 | 0.4379 s |
| move the cube to the red circle | 0.2000 → 0.2000 | 否 | -1.724 | 0.4359 s |
| push the block onto the red goal | 0.2000 → 0.2000 | 否 | -1.832 | 0.4315 s |

- **基础成功率：0/3 = 0.0**
- 平均单步耗时 0.4351 s，总耗时 125.27 s，显存保留 4.823 GB

逐步动作分析（第一条指令 30 步）：x 方向为正的步数 20/30，方向大体正确；动作范数均值 0.010538 m、最大 0.018669 m，小于专家控制律每步 0.02 m 的量程；末端有位移但立方体始终未被推动。

### 4.3 结论与边界

- 这是**有效图像输入下的零样本基线**，成功率 0 属于预期现象：预训练 VLA 主要基于真实世界数据，本仿真场景存在明显域差距，且末端未建立接触。
- 该数字只用于微调前后对照，不代表管线错误；单样本推理已证明模型加载、动作映射与环境步进链路无报错。
- 不做任何“任务成功率提升”的推断；后续需用自有数据微调后重测同一基线。

## 5. 复现命令（vla 环境）

```bash
# 单样本推理（使用已保存图像与状态）
python vla/inference/inference.py \
  --model-dir /data/wzh/robot/models/openvla-7b --quant int4 \
  --image results/vla_single_step/observation_224.png \
  --state results/vla_single_step/initial_state.npz \
  --instruction "push the cube to the red target"

# 实时取图必须先设置 EGL
MUJOCO_GL=egl python vla/inference/inference.py \
  --quant int4 --instruction "push the cube to the red target"

# 零样本闭环基线
MUJOCO_GL=egl python vla/inference/rollout_vla_closed_loop.py \
  --quant int4 --max-steps 30 \
  --out-dir results/vla_closed_loop_int4_egl
```
