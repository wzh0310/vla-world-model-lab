# VLA 指令与场景泛化数据集 · 统计报告

生成位置：`/data/wzh/robot/datasets/vla_pushcube_finetune/hf_dataset/`（数据不入 git）。
以下数字均来自容器内 `dataset_summary.json` / `manifest.json` / 校验结果，非估算。

## 数据来源

- 基础轨迹：既有 PushCube 数据集 25 条（物理空间 4 维动作、224×224 RGB）。
- 变体轨迹：左右目标各 10 条，用 `PushCubeEnv(scene_override=True)` 注入目标坐标采集。

| 变体 | 目标 XY | 采集数 / 尝试数 | 最终距离上界 | 帧数区间 | 拒绝原因 |
| --- | --- | --- | --- | --- | --- |
| 左目标 | (0.82, -0.02) | 10 / 24 | 0.0233 m | 35–40 | 未到达 6 / 帧数 6 / 穿模 2 |
| 右目标 | (0.82, -0.22) | 10 / 41 | 0.0268 m | 36–40 | 未到达 28 / 帧数 1 / 穿模 2 |

## 规模与划分

按物理轨迹分组、类别分层、seed=42 做 8:1:1 划分。

| split | 物理轨迹 | 语言条件化轨迹 | 帧数 |
| --- | --- | --- | --- |
| train | 31 | 109 | 4044 |
| val | 7 | 23 | 882 |
| test | 7 | 23 | 858 |
| 合计 | 45 | 155 | 5784 |

指令类别 7 类：direct / brief / descriptive / variant / reverse / variant_left /
variant_right。三类 split 均覆盖全部 7 类；帧级分布中变体合计约 59%（每条变体
4 条指令且帧数偏多），为指令展开后的真实分布，未人为再平衡。

## 动作归一化

全量轨迹 min-max → `[-1,1]`，常量维置 0：

- min = `[0.0, -0.00962, -0.00318, 0.0]`
- max = `[0.02, 0.008, 0.0, 0.0]`
- 常量维 = `[3]`（gripper 恒 0）

参数见 `hf_dataset/norm_stats.json`，微调与推理必须复用同一份。

## 字段

`image`（Image RGB 224×224）、`instruction`（string）、`action`（float32 (4,) ∈ [-1,1]）。

## 校验

- 数据集校验：23/23 通过（字段、图像尺寸、动作范围、指令类别、轨迹规模、无跨集泄漏）。
- 适配器 + collator 校验：7/7 通过（batch 形状、prompt 格式、动作 token 监督位）。
  使用 mock processor，不加载 7B 模型，仅验证接口，**不构成训练收敛或成功率结论**。

详见 `docs/VLA指令与场景泛化数据集构建.md`。
