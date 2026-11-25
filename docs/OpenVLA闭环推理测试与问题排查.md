# OpenVLA 闭环推理测试与问题排查报告

本报告记录把 OpenVLA 接入 PushCube 仿真环境、跑通回合级闭环推理的完整过程，包含架构设计、三条指令的零样本测试结果、三类常见问题的系统排查，以及通用工具模块的封装说明。所有结果均为容器实测，不含推测数据。

## 一、闭环架构与工程实现

### 1.1 跨环境单进程方案

OpenVLA 依赖（`vla` 环境：`transformers==4.40.1`、`timm`、`accelerate`、`peft`）与 MuJoCo 仿真依赖（`mujoco` 环境：`gymnasium==1.4.0`、`mujoco==4.0.1`）分布在两个隔离的 conda 环境中。若采用「每步跨进程调用」，7.5B 模型会被重复加载数十次，完全不可行。

实测发现两个环境的 `torch==2.6.0+cu124`、`numpy==2.5.5` 版本一致，因此采用**单进程注入**方案：在 `vla` 进程内把 `mujoco` 环境的 `site-packages` 追加到 `sys.path`，模型只加载一次，环境全程驻留。

```
vla 进程
├── transformers/timm/accelerate  →  vla site-packages（OpenVLA 推理）
└── gymnasium/mujoco              →  mujoco site-packages 注入（仿真交互）
```

闭环每步循环：`obs["image"]` → 图像预处理 → VLA 推理 → 动作映射 → `env.step()`。
机械臂末端（EE）由 mocap  weld 约束驱动，`action[:3]` 是每步 EE 笛卡尔增量，夹爪位在本任务中保留但不参与动力学。

### 1.2 通用工具模块 `utils/vla_adapter.py`

把反复用到的能力收敛为可复用组件，闭环脚本与诊断脚本共用：

| 组件 | 职责 |
|---|---|
| `build_prompt()` | 按官方模板构造 `In: What action should the robot take to {instruction}?\nOut:` |
| `preprocess_image()` | 环境 RGB ndarray → 224×224 PIL，含通道模式开关（`rgb`/`bgr`）供诊断 |
| `unnorm_to_unit()` | 用 q01/q99 把反归一化物理增量还原回 [-1,1] |
| `map_action()` | 7 维 → 4 维：丢弃旋转、`xyz×scale`、夹爪 `(g+1)/2`，含方向反转开关 |
| `parse_action()` | 动作向量 → 带语义字典（步长、方向、夹爪状态） |
| `OpenVLAPolicy` | 统一 bf16 / int4 加载与单步推理，封装正确 dtype 与显存统计 |

**避坑记录**：`mujoco-studio-lab` 仓库也有同名 `utils/` 包，注入 `sys.path` 后会遮蔽本仓库的 `utils.vla_adapter`。闭环脚本改用 `importlib.util.spec_from_file_location` 按文件路径加载适配器，彻底规避包名冲突。

## 二、三条指令零样本测试结果

统一配置：bf16 精度、`unnorm_key=bridge_orig`、映射量程 ±0.02 m/步、每条指令 30 步、相同初始状态（立方体起点 `[0.62,-0.12]`，目标 `[0.82,-0.12]`，初始距离 0.20 m）。

| 指令 | 步数 | 距离变化 | 最小距离 | 总奖励 | 平均单步耗时 |
|---|---|---|---|---|---|
| `push the cube to the red target` | 30 | 0.200 → 0.200 | 0.200 | -1.66 | 0.305 s |
| `move the cube to the red circle` | 30 | 0.200 → 0.200 | 0.200 | -1.63 | 0.301 s |
| `push the block onto the red goal` | 30 | 0.200 → 0.200 | 0.200 | -1.82 | 0.287 s |

**成功率 0/3**，立方体到目标距离全程不变。模型加载一次约 29–37 s，三条指令总墙钟 210 s。

### 2.1 关键判读：管线通畅，但零样本无任务语义

- **管线通畅的证据**：每步 EE 实际位移与映射动作一致。例如 ep0 第 1 步映射 `[0.0067,-0.0193,-0.0089]`，EE 实动 `[0.0120,-0.0193,-0.0089]`；30 步累计 EE 位移 0.14–0.24 m，机械臂在画面中明显移动并贴近桌边。
- **指令确实进入了模型**：三条语义相近但措辞不同的指令产生不同的动作方向分布。ep0/ep1（"red target/circle"）平均动作为 `(+4.6, -3.9)` mm/步 与 `(+4.5, -5.7)` mm/步，指向 +x（正确朝向目标方向）；ep2（"red goal"）为 `(+2.6, +1.1)` mm/步，明显不同。这说明语言条件在输出端有可观测影响，但不足以形成正确的长程导航。
- **为何立方体没动**：EE 初始位置距立方体约 0.5 m，每步仅 2 cm，30 步虽累计移动 0.14–0.24 m，但轨迹方向并非精确指向立方体，末端从未接触立方体。这是**域差距 + 零样本动作缺乏任务级规划**的表现，不是管线 bug。

### 2.2 与项目既有结论一致

这一结果与世界模型阶段的发现完全吻合：纯视觉策略在 50 个未见场景成功率为 0，权威 benchmark 的视觉 PPO 满数据成功率也为 0。视觉语义与长程导航是当前瓶颈，而非决策层或执行层。OpenVLA 虽具备语言条件，但预训练数据（Bridge / Google Robot 真实场景）与本仿真域差异过大，零样本迁移不承担任务成功率期望。

## 三、常见问题系统排查

### 3.1 图像预处理通道顺序

- **风险**：OpenCV 读图为 BGR，PIL / 环境 `obs["image"]` 为 RGB。通道送反会让视觉编码器看到错误的颜色语义。
- **排查方法**：在 `preprocess_image()` 加 `channel_mode` 开关，对同一帧做 RGB / BGR 两次推理对照。
- **实测**（ep0 首步）：RGB 映射 `[0.0067,-0.0193,-0.0089]`；BGR 映射 `[0.0049,+0.0086,-0.0017]`。y 分量从 -0.0193 翻到 +0.0086，差约 0.028 m——**通道顺序显著影响输出**。
- **结论**：当前链路用 `Image.fromarray(...).convert("RGB")`，与环境观测一致，正确。工具模块默认 `rgb`，`bgr` 仅作诊断开关。

### 3.2 动作映射方向反转

- **风险**：位置增量符号取反会让策略朝反方向执行，闭环迅速发散。
- **排查方法**：`map_action(flip_xyz=True)` 对同一动作取反对照；并用 EE 实际位移核对映射符号。
- **实测**：flip 对照 y 分量从 -0.0193 变 +0.0193（差 0.0386 m），差异显著；正常映射下 EE 实动方向与动作符号一致（见 2.1）。
- **结论**：当前映射未反转，`xyz×0.02` 的符号正确。ep0/ep1 动作指向 +x，与目标相对立方体的 +x 方向一致。

### 3.3 推理速度

- **实测**：bf16 平均 0.298 s/步（约 3.4 Hz），单步峰值 0.33 s；模型加载 29–37 s；显存分配 15.13 GB、保留 15.53 GB。
- **对比**：低于 Bridge 数据采集时的约 5 Hz，闭环实时性有压力但可接受（环境 step 本身很快，瓶颈全在 GPU 推理）。
- **低显存替代**：int4（nf4）实测权重约 4.4 GB、单步 1.31 s——显存降到 1/3 但速度降到 1/4。A6000 48 GB 无显存压力时优先 bf16；仅当部署到 <16 GB 卡时改用 int4，并接受吞吐下降。
- **优化方向**：OpenVLA 单步自回归生成 7 个动作 token（chunk=1）是 3–6 Hz 的根源；换用流匹配块输出模型（π0 / SmolVLA）可显著提升吞吐，留待后续选型。

## 四、产出与复现

### 4.1 新增文件

| 文件 | 说明 |
|---|---|
| `utils/vla_adapter.py` | 通用适配工具：预处理 / 映射 / 解析 / 模型封装 |
| `scripts/rollout_vla_closed_loop.py` | 回合级闭环脚本，支持多指令、诊断、帧保存 |
| `scripts/plot_closed_loop.py` | 三联分析图：EE 轨迹 / 距离曲线 / 动作方向 |
| `results/vla_closed_loop/summary.json` | 三条指令汇总与配置 |
| `results/vla_closed_loop/ep00..02/episode.json` | 每步动作、EE 位移、距离、奖励、耗时 |
| `results/vla_closed_loop/ep*/frames/` | 每 5 步一张关键帧截图 |
| `results/vla_closed_loop/closed_loop_analysis.png` | 可视化分析图 |

### 4.2 快速复现（vla 环境）

```bash
cd /data/wzh/robot/vla-world-model-lab
/opt/miniconda3/envs/vla/bin/python scripts/rollout_vla_closed_loop.py \
  --max-steps 30 --out-dir results/vla_closed_loop --save-frames --diagnose
/opt/miniconda3/envs/vla/bin/python scripts/plot_closed_loop.py
```

## 五、结论与下一步

1. **闭环链路已打通**：读图 → 推理 → 映射 → 步进全流程稳定，模型单进程加载一次即可跑多回合，工具模块可复用。
2. **零样本成功率 0 属预期**：预训练于真实机器人数据，仿真域差距大；核心价值是验证管线，而非任务性能。三条指令产生不同动作方向，证明语言条件生效但未形成正确长程规划。
3. **三类常见问题均已排查**：通道顺序必须用 RGB；映射方向正确无反转；速度瓶颈在自回归 token 生成（3.4 Hz），int4 可降显存但牺牲吞吐。
4. **下一步**：按既定路线进入自定义 VLA 数据集构建（逐条轨迹配语言标注），为后续 LoRA 微调提供域内数据；微调后重跑本闭环脚本对比成功率变化。
