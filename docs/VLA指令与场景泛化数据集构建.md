# VLA 指令与场景泛化数据集构建与标准化

本阶段为 OpenVLA LoRA 微调准备标准化数据集：在既有 25 条 PushCube 基础轨迹之上，
新增「推到目标区偏左 / 偏右」两类空间变体，把每条物理轨迹展开为多条语言条件化
轨迹，统一为 HuggingFace Dataset 格式并按 8:1:1 划分。所有数据落在仓库外
`/data/wzh/robot/datasets/vla_pushcube_finetune/`，脚本在 `vla_finetune/data/`。

## 一、环境分工与版本兼容

| 环境 | 职责 | 关键包 |
| --- | --- | --- |
| `mujoco` | 变体轨迹采集（EGL 离屏渲染） | mujoco 4.0.1 / torch 2.6+cu124 |
| `vla` | HF Dataset 生成、校验、微调消费 | datasets 3.6.0 / peft 0.11.1 |
| `lerobot` | LeRobot 格式接口验证 | lerobot 0.7.0 / datasets 5.2.0 |

踩坑记录：`lerobot` 的 datasets 5.2.0 把定长序列字段序列化为 `_type: List`，
而 `vla` 的 datasets 3.6.0 只识别 `Sequence`，直接用 5.2.0 落盘的数据集在
微调端加载报 `Feature type 'List' not found`。由于数据集最终在 `vla` 环境消费，
**最终 HF Dataset 统一用 vla 环境（datasets 3.6.0）生成**，保证落盘格式与
消费端版本一致；`lerobot` 环境仍保留其既有 LeRobot 格式验证职责，不混用。

## 二、任务变体设计

基础任务目标 `TARGET_XY=(0.82,-0.12)`。两个变体通过 `PushCubeEnv(
scene_override=True)` 在 reset 时注入新的 `target_xy`，真实移动桌面红色圆盘：

| 变体 | 目标 XY | 类别 | 采集数 | 尝试数 | 最终距离上界 | 帧数区间 |
| --- | --- | --- | --- | --- | --- | --- |
| 左目标 | (0.82, -0.02) | `variant_left` | 10 | 24 | 0.0233 m | 35–40 |
| 右目标 | (0.82, -0.22) | `variant_right` | 10 | 41 | 0.0268 m | 36–40 |

筛选判据与基础采集一致：到达目标（dist < 0.03）、帧数 30–40、无穿模/瞬移。
立方体起点 y 贴近各自目标 y（±0.03 m）以保证横向修正可控；红盘位置与指令语义
仍真实改变，场景/语言差异保留。每个变体配 4 条同义指令（如 "push the cube to
the left side of the red target" / "move the cube to the left part of the red
circle" 等），措辞只引用场景已有物体。

## 三、指令展开与划分

- 基础 25 条：每条 = 原模板指令 + 2 条同义改写（复用既有 `PARAPHRASES`）= 3 种表述。
- 变体 20 条：每条 4 条同义指令。
- 合计 **45 条物理轨迹 → 155 条语言条件化轨迹**。

动作用全量轨迹 min-max 归一化到 `[-1,1]`：`x_norm = 2*(x-min)/(max-min)-1`，
常量维（gripper 恒 0）置 0。参数落在 `hf_dataset/norm_stats.json`，微调与推理
复用同一份，勿与 OpenVLA 自带 `unnorm_key` 统计量混用。

划分按**物理轨迹分组**做分层 8:1:1：同一物理轨迹的所有指令表述整体进入同一
split，杜绝跨集泄漏；按类别分层保证分布一致，seed=42 可复现。

| split | 物理轨迹 | 语言条件化轨迹 | 帧数 |
| --- | --- | --- | --- |
| train | 31 | 109 | 4044 |
| val | 7 | 23 | 882 |
| test | 7 | 23 | 858 |

七类指令（direct / brief / descriptive / variant / reverse / variant_left /
variant_right）在三个 split 均有覆盖。注意变体每条 4 条指令且帧数偏多，帧级
分布中变体占比约 59%；这是指令展开后的真实分布，已在 `dataset_summary.json`
按类别逐 split 记录，未做人为再平衡。

## 四、字段与加载

| 字段 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- |
| `image` | Image (RGB) | 224×224×3 | 顶视相机帧，与 OpenVLA 输入分辨率对齐 |
| `instruction` | string | 标量 | 回合级自然语言指令 |
| `action` | float32 | (4,) | `[dx, dy, dz, gripper]`，归一化到 [-1,1] |

`hf_dataset_adapter.py` 把 HF 样本包装为 PyTorch Dataset，产出 `image`
`[3,224,224]` float、`prompt`（完整 OpenVLA 输入文本）、`action` `[4]` float，
直接对接 `src/vla_finetune/data.py` 的 `FinetuneCollator`；4 维动作在 collator
内按 `[dx,dy,dz,0,0,0,gripper]` 扩为 7 维，与既有 LoRA 骨架口径一致。

## 五、校验结论

- 数据集校验 **23/23 通过**：三 split 字段/图像尺寸/动作范围/指令类别/轨迹规模/
  无跨集泄漏全部满足（结果 `verify_result.json`）。
- 适配器 + collator 校验 **7/7 通过**：batch 形状、prompt 格式、动作 token 监督位
  正确（结果 `adapter_verify_result.json`）；此项用 mock processor，**不加载 7B
  模型**，仅验证数据与训练代码接口，不构成训练收敛或任务成功率结论。
- 语言条件化轨迹 155 ≥ 100、物理轨迹 45、类别 7 ≥ 3，达到本阶段数据目标。
