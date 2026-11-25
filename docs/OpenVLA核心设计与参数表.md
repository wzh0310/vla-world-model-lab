# OpenVLA 核心设计与参数表

> 目标：吃透 OpenVLA 的输入输出规范与三个关键设计（动作 Token 化 / 跨机器人归一化 /
> LoRA 小样本微调），为 PushCube 场景适配列出硬约束。
> 所有模型卡参数以 HuggingFace `openvla/openvla-7b` 与官方 GitHub 仓库为准（文末链接）。

---

## 一、模型总体结构（对应 VLA 四段式）

| 段 | OpenVLA 实现 | 关键参数 |
|---|---|---|
| ① 视觉编码器 | **DINOv2 ViT-L/14 + SigLIP SoViT-400M/14 双塔**，特征融合 | 输入 224×224 RGB；本体状态 q_t 经 FiLM 注入 DINOv2 特征 |
| ② 多模态对齐层 | 线性投影，把视觉特征映射到 LLM 词表空间（Prismatic VLM 范式） | — |
| ③ 语言主干 | **Llama 2 7B**（总参数约 7.6B） | 词表扩展 +256 个动作 token |
| ④ 动作解码头 | 自回归生成 **7 个动作 token**，再查表反离散化 + 反归一化 | 每维 256 bin；单步输出（chunk=1） |

---

## 二、关键设计 1：动作 Token 化

**动作向量（7 维，均为增量 ΔEE）**：

```
a = [ Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper ]
      └─ 位移增量(米) ─┘  └─ 轴角旋转增量 ─┘  └─ 夹爪
```

**离散化流程**（训练与推理对称）：

```
连续动作 a_j
   → 按数据集统计归一化到 [-1, 1]（见设计 2）
   → 每维均匀切 256 个 bin：bin = round((a_norm + 1) / 2 × 255)
   → 查表映射为词表中 256 个专用动作 token（起始 ID 31744，见官方仓库 tokenizer）
   → 7 个 token 按 [Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper] 固定顺序拼在 prompt 后
推理：采样 7 个 token → 反查 bin 中心 → 反归一化 → 7 维连续动作
```

要点与约束：
- 训练目标是**标准下一 token 交叉熵**，控制问题被完全同构成生成问题；
- 离散化粒度 256 bin 是精度上限来源之一（对 Δpos ≈ 米级单位，bin 宽约 (q99-q01)/255，
  对厘米级增量任务足够，但**高频精细控制会受量化损失**）；
- 单步输出（chunk=1）+ 生成 7 个 token → 推理频率约 **6 Hz（RTX 4090 bf16）**，
  远低于 π0/SmolVLA 的流匹配块输出，这是自回归路线的核心代价；
- 贪心解码（`do_sample=False`）是官方推荐，采样会带来动作抖动。

---

## 三、关键设计 2：跨机器人动作归一化

不同机器人的动作空间不同，OpenVLA 用**统一 7 维 ΔEE 表达 + 数据集级统计归一化**解决：

1. **动作空间统一**：预训练混合两个数据集家族，全部表达为 7 维末端增量——
   Google Robot（Fractal）7 自由度、BridgeData V2（WidowX）6 自由度 + 夹爪，
   缺失维度补齐，形成统一 token 布局；
2. **数据集级统计归一化**：每维用该数据集自身的 **q01 / q99 分位数**边界归一化到
   [-1, 1]，统计量随 checkpoint 存放在 `config.json` 的 `norm_stats`；
3. **维度 mask**：`norm_stats[key]["action"]["mask"]` 标记哪些维度参与反归一化
   （例如 `bridge_orig` 共 7 维、最后一维夹爪为 False——夹爪走独立的二值化路径）；
4. **夹爪二值化**：训练/微调时夹爪动作统一二值化为 {0, 1}，避免不同本体
   连续/离散夹爪语义混乱。

**使用约束（最容易翻车的点）**：
- 反归一化必须用 `predict_action(**inputs, unnorm_key="bridge_orig")` 显式指定统计键；
  **拿错键 = 输出尺度全错**，机器人会朝错误方向猛冲；
- `unnorm_key` 不是通用参数：bridge_orig 只对应 Bridge 统计；微调自己的数据后
  **必须换成自己的统计键**；
- `predict_action` 返回值**已经反归一化**，不要再做一次尺度变换；
- 适配新场景时把自己的数据集统计量写进 `norm_stats`（键名自定义，如 `pushcube_orig`），
  并同步 q01/q99 与 mask。

---

## 四、关键设计 3：LoRA 小样本微调

| 项 | 官方默认（GitHub `vla-scripts/finetune.py`） | 说明 |
|---|---|---|
| LoRA rank | **32** | 论文报告 rank32 接近全量微调，且分布外泛化更好 |
| 目标模块 | LLM 全部线性层（attention q/k/v/o + MLP） | 视觉塔默认不动 |
| 可训练参数占比 | 约 **1.4%** | 显存与数据需求大幅下降 |
| 学习率 | **5e-4** | LoRA 专用较大学习率 |
| batch size | 16（+ `grad_accumulation_steps` 累积） | 官方示例 batch16 约 72GB；小卡需减小 batch + 累积梯度 |
| 数据增强 | `--image_aug True`（随机裁剪等） | 对小样本适配重要 |
| 量化选项 | 4-bit 量化 + LoRA 可进一步压显存（论文演示单张消费级卡可跑） | bf16 LoRA 下限约 >24GB |
| 数据格式 | RLDS（Open X-Embodiment 管线） | 自定义数据需转 RLDS 或写 PyTorch 包装 |

论文结论：LoRA 微调使 OpenVLA 能用**少量演示**适配新机器人/新场景
（跨本体 ALOHA、Franka 实验各用数十条演示），这是本项目下阶段
「少量标注轨迹 → LoRA 适配」的可行性依据。

---

## 五、官方 HuggingFace 模型卡参数对照表（输入输出规范）

| 项 | 值 | 备注 |
|---|---|---|
| 检查点 | `openvla/openvla-7b` | 许可 MIT |
| 输入图像 | **单机位 RGB，224×224** | 第三人称工作区视角；processor 内部完成 resize/归一化 |
| 指令 | 英文自然语言 | 裸指令是分布外输入，**必须套官方 prompt 模板** |
| prompt 模板 | `In: What action should the robot take to {INSTRUCTION}?\nOut:` | 逐字固定，含 `In:`/`Out:` 前后缀 |
| 本体状态 | 可选，FiLM 注入视觉特征 | 基础模型卡未强制 |
| 输出动作 | **7 维 ΔEE 增量**（Δxyz 位移 + Δ轴角旋转 + gripper） | 不是 7 关节角 |
| 归一化范围 | 各维 [-1, 1]（数据集 q01/q99 统计），反归一化由 `unnorm_key` 驱动 | `config.json → norm_stats` |
| 输出频率 | 约 6 Hz（RTX 4090 bf16；A100 4.2 Hz） | 自回归单步的固有延迟 |
| 解码方式 | 贪心（`do_sample=False`） | 采样会抖动 |
| 依赖版本 | `transformers==4.40.1`、`tokenizers==0.19.1`、`timm==0.9.10`（requirements-min） | 版本敏感，需隔离环境 |
| 零样本边界 | 模型卡明确：仅限训练数据中出现过的机器人/域组合 | 未见本体/视角必须微调 |

**官方最小推理代码**（`vla` 环境直接可用）：

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

MODEL_ID = "openvla/openvla-7b"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16,        # 必须用 bf16，fp16 明显掉精度
    low_cpu_mem_usage=True, trust_remote_code=True,
).to("cuda:0")

image = Image.open("front_cam.jpg").convert("RGB")
instruction = "push the red cube to the green target"
prompt = f"In: What action should the robot take to {instruction}?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
# action: 7 维 numpy（已按该统计键反归一化）
```

---

## 六、PushCube 适配约束表（后续微调的硬约束清单）

| 约束项 | OpenVLA 要求 | PushCube 现状 | 适配方案 |
|---|---|---|---|
| 图像输入 | 单机位 RGB 224×224 | 224×224 原生 | ✅ 直接用，固定机位与训练一致 |
| prompt | 官方模板 + 英文指令 | 无语言标注 | 逐条轨迹补英文指令（模板化生成） |
| 动作维度 | 7 维 ΔEE（Δxyz + Δ轴角 + gripper） | 4 维 [dx, dy, dz, gripper] | Δxyz 直接映射；Δ轴角置 0；gripper 放第 7 维 |
| 夹爪语义 | 二值化 {0,1} | 连续/离散开合 | 二值化写入数据集统计 |
| 归一化 | 数据集 q01/q99 + mask + 自定义 unnorm_key | 无 | 生成 `norm_stats["pushcube_orig"]`（q01/q99 按数据集实测） |
| 坐标系 | ΔEE 相对机器人基座/末端约定 | env 动作即末端位移增量 | 需写清单轴方向与单位（米），首跑空载验证方向 |
| 控制频率 | 模型约 6 Hz | 仿真无实时约束 | 可接受；每回合 50 步离线评估 |
| 显存 | bf16 LoRA >24GB；batch16 约 72GB | A6000 48GB | batch 减半 + 梯度累积，或 4-bit 量化 LoRA |
| 依赖 | transformers 4.40.1 / timm 0.9.10 | vla 环境为 torch 2.6 + 新版 transformers | **建议独立 conda 环境**（如 `openvla`），避免破坏现有 vla 环境 |
| 数据格式 | RLDS / PyTorch Dataset 包装 | LeRobotDataset + HDF5 | 写包装类输出 (image, instruction, action) 三元组 |
| 零样本边界 | 仅限训练域组合 | 全新仿真域 | 不指望零样本成功，直接走 LoRA 微调路线 |

---

## 七、常见坑清单（来自官方示例与社区复盘）

1. **prompt 必须逐字用官方模板**：去掉 `In:`/`Out:` 会显著掉成功率（分布外输入）；
2. **fp16 掉精度**：必须 `torch.bfloat16`；输入张量要与模型 dtype 对齐，否则报类型错误；
3. **unnorm_key 错配**：输出尺度全错，是"模型在动但方向不对"的最常见根因；
4. **重复反归一化**：`predict_action` 已反归一化，再变换会缩小尺度；
5. **动作是 ΔEE 不是关节角**：下游控制器要自己做坐标系/单位/限位换算；
6. **首次上环境先空载验证**：打印动作序列的方向与幅值，确认映射后再真跑；
7. **token 准确率 ≠ 任务成功率**：离线 token accuracy 高也可能 rollout 失败，
   评估必须闭环（这一点与世界模型想象策略评估的教训完全一致）。

---

## 八、参考

- HuggingFace 模型卡：https://huggingface.co/openvla/openvla-7b
- 官方仓库与微调脚本：https://github.com/openvla/openvla
- 论文：OpenVLA: An Open-Source Vision-Language-Action Model（arXiv:2406.09246，CoRL 2024）
- OpenVLA-OFT（连续动作头 + 并行解码改进）：https://github.com/moojink/openvla-oft
- 适配注意事项复盘（2026-10 核对）：
  https://physicalailab.io/en/openvla-getting-started-input-output-hardware
