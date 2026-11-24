# vla-world-model-lab

具身智能世界模型实验室：从离线交互轨迹学习「图像 + 动作 → 隐状态 → 下一帧图像」的动力学模型，
为后续 VLA 策略的想象 rollout / 数据增强 / 规划提供基础。**全程在 `vla` conda 环境开发。**

> **项目范围说明**：本项目聚焦**算法层世界模型的实现与验证**——RSSM 动力学建模、
> 多步开环想象、奖励/价值头与想象策略链路，全部基于 **MuJoCo 仿真环境**（PushCube-v0 推立方体）
> 的离线轨迹验证；**不涉及底层机器人控制**（电机驱动、实机部署、运动学/动力学求解等均不在范围内）。
> 真实机器人迁移属于后续工作。

当前阶段：**多步开环想象 + 消融实验完成，世界模型核心链路闭环可用** —— 已完成视觉自编码器、RSSM 动力学、三损失联合训练，
并在统一评估集上完成 PSNR / SSIM / 奖励误差的量化验证与连续 10 步预测可视化。

---

## 一、当前进度

| 模块 | 状态 | 说明 |
|---|---|---|
| 离线轨迹数据底座 | ✅ | 100 条轨迹（40 专家 + 60 随机），HDF5，约 70MB，不入 git |
| 混合策略数据扩充 `scripts/collect_mixed_trajectories.py` | ✅ | PPO + 随机策略采集 100 条/5000 帧，初始位置/幅度多样化 |
| 标准化格式化与 9:1 划分 `scripts/format_dataset.py` | ✅ | (obs,action,reward,next_obs,done) 标准格式，train/test 无泄漏 |
| 轻量图像增强（平移+亮度+随机裁剪） | ✅ | 整段序列共享参数；扩充数据集上验证有效，全量训练开启 |
| 全量训练 `scripts/train_full.sh` | ✅ | 分组学习率(enc 1e-4/dyn 3e-4) + 序列长度 10→20 课程 + 每 1000 步评估存最优 |
| 序列数据集 `datasets/episodic_dataset.py` | ✅ | 20 步序列 / 64×64 / 动作与奖励归一化 |
| GPU 显存预加载采样 `datasets/fast_loader.py` | ✅ | 约 80× 加速（1000 步从 ~70min → 51s） |
| 视觉自编码器 `models/vae.py` | ✅ | 兼容转发层；实现已拆分为 encoder.py / decoder.py |
| 模块化拆分 `models/encoder.py` `decoder.py` `reward_predictor.py` | ✅ | 编码器/解码器/预测头各自独立，权重 key 完全兼容旧检查点 |
| 模型工厂 `models/factory.py` | ✅ | 按配置构建动力学与预测头，统一 RSSM/MLP 接口 |
| RSSM 动力学 `models/rssm.py` | ✅ | 确定性态 200 + 随机态可配，特征维度随 stoch 变化 |
| 纯 MLP 动力学基线 `models/mlp_dynamics.py` | ✅ | 无循环/无随机的前馈对照，用于长程预测动力学消融 |
| 四损失联合训练 `trainer/train_world_model.py` | ✅ | 重建 + KL(free bits) + 奖励 + 价值，4000 步收敛 |
| 单步预测全面验证 `trainer/eval_single_step.py` | ✅ | PSNR / SSIM / 奖励 MAE·RMSE，统一评估集 |
| 预测可视化 `trainer/make_prediction_gif.py` | ✅ | 连续 10 步真实 vs 单步预测对比 GIF |
| 开环多步预测 `models/rssm.py` `imagine_rollout` | ✅ | 仅首帧+动作，全程不接真实图像，任意步长 |
| 奖励头 / 价值头 `models/reward_predictor.py`（`heads.py` 兼容转发） | ✅ | 奖励单步回归 + 状态价值（折扣回报监督） |
| 奖励/价值曲线验证 `trainer/verify_heads.py` | ✅ | 奖励 corr 0.77，价值 corr 0.55（含局限说明） |
| 多步预测调优（KL 权重） | ✅ | free_bits 0.05+kl_scale 1.5，10步开环 22.39→24.71dB |
| 多步基准评估 `trainer/eval_multistep.py` | ✅ | 多轨迹逐 step PSNR/SSIM，统一对比口径 |
| 多步长全面评估 `trainer/eval_multistep_full.py` | ✅ | 1/5/10/20 步 PSNR 衰减 + 各步长奖励误差 + 折线图 + 3 组轨迹视频 |
| 极简策略网络 `models/policy.py` | ✅ | MLP 隐状态→动作（tanh 限幅），约 10 万参数 |
| 隐空间随机策略 Actor-Critic `models/actor_critic.py` | ✅ | 高斯 Actor（μ+log_std 重参数化）+ 价值 Critic，严格对齐隐维度，动作映射回真实范围，26 万参数 |
| 想象策略网络验证 `trainer/verify_actor_critic.py` | ✅ | 随机策略在 10 步纯想象 rollout 正常输出动作，梯度沿奖励链可回传 |
| 想象轨迹训练 `scripts/train_imagine_policy.py` | ✅ | Dreamer 核心：λ-return + PPO + 熵正则 + Huber 价值损失，冻结世界模型仅更新策略，固定探针选最优；v2 想象回报 -0.04→+0.85 |
| 真实环境零样本评估 `scripts/eval_imagine_policy.py` | ✅ | mujoco 环境 20 回合；成功率 0%（专家 ACT 基线 30%），sim2real 间隙与长程导航瓶颈诚实记录 |
| 想象策略训练 Demo `trainer/imagine_policy.py` | ✅ | 世界模型内想象→奖励回报→可微反传更新策略，回报 1.32→2.59 上升 |
| 达标验证 `scripts/verify_targets.py` | ✅ | 测试集 20 步 SSIM 0.7511≥0.6，奖励 MAE 0.024<0.5，全部达标 |
| 统一配置 `configs/default.yaml` | ✅ | data / model / training / eval 四段，命令行优先 |
| 消融实验（stoch 维度 + KL 有无） | ✅ | 4 组对照，1/5/10/20 步评估，表格+折线图+稳定性柱状图 |
| 一键推理 `trainer/infer_video.py` | ✅ | 动作序列 → 开环预测视频（轨迹/JSON/随机三种来源） |
| 配置支持分辨率/序列长/隐维度可配 | ✅ | VAE 自适应池化+resize 向后兼容改造，64 分辨率权重兼容 |
| 统一推理封装 `models/world_model.py`（rollout 管线） | ✅ | 一键加载检查点；前 K 帧后验热身 + 开环想象，同步输出预测图像/隐状态/奖励/价值序列 |
| 四级评估指标体系 + 统一评估入口 `scripts/eval_world_model.py` | ✅ | 像素/结构/感知/动力学四级指标，一条命令输出全部指标+精度-步长衰减曲线 |
| KL 权重退火（--kl-warmup）与对照实验 | ✅ | step1 +0.84dB 但长程无收益，确认 free bits 已主导防坍缩（诚实记录） |
| 多步预测对比可视化 `scripts/compare_video.py` | ✅ | 3 条测试轨迹真实vs预测并排视频，标注 5/10/20 步衰减，入库 results/ |
| 统一消融评估 `scripts/eval_ablation_unified.py` | ✅ | 同一数据/配置下维度(16/32/48/64) + 动力学(RSSM/MLP)对照，图表指标入 results/ |
| 完整 DreamerV2 闭环（λ-return + 价值引导 + 策略蒸馏） | ⬜ | 下一阶段目标 |

---

## 二、项目整体架构与算法原理

### 2.1 端到端数据流（文字版架构图）

```
【数据层】MuJoCo Panda 仿真 (mujoco-studio-lab)
    PushCube-v0 推立方体：100 条轨迹（40 专家 scripted + 60 随机），224×224 RGB
        │ scripts/export_trajectories.py（EGL 离屏渲染）
        ▼
    outputs/data/trajectories.h5（图像/动作/奖励，不入 git）
        │ datasets/fast_loader.py：GPU 显存预加载 + 20 步序列采样（约 80× 加速）
        ▼
【模型层】模块化世界模型（models/，已解耦为独立文件）
    encoder.py           ConvEncoder: 64×64×3 → 观测嵌入 e_t (64)
    rssm.py              RSSM: h_t = GRU([z_{t-1},a_t], h_{t-1})，后验 q(z|h,e) / 先验 p(z|h)
    mlp_dynamics.py      MLPDynamics: 纯前馈动力学基线（无循环、无随机），消融对照
    decoder.py           ConvDecoder: feature [h,z] → 64×64×3 预测帧
    reward_predictor.py  RewardHead / ValueHead: feature → 奖励 / 价值
    factory.py           模型工厂：按配置构建动力学与预测头，统一 RSSM/MLP 接口
    world_model.py       统一推理封装（rollout 管线）
        │ vae.py / heads.py 为向后兼容的转发层；权重 key 完全不变
        │ trainer/train_world_model.py：重建 + KL(free bits) + 奖励 + 价值 四损失
        ▼
【评估与想象层】（trainer/）
    eval_single_step      单步三口径 PSNR/SSIM + 奖励误差
    eval_multistep_full   1/5/10/20 步开环 PSNR 衰减曲线 + 视频
    imagine_policy        世界模型内想象轨迹 → 可微反传更新策略（Demo）
    infer_video           一键推理：动作序列 → 预测视频
        ▼
【下游衔接】embodied-policy-zoo（BC/ACT/PPO/IQL/DT 策略库）可消费想象数据
```

### 2.2 核心算法原理（精简版，详见 docs/）

- **为什么要世界模型**：让智能体在"脑内"推演未来——给定当前观测与候选动作序列，
  不与真实环境交互即可预测后续图像与奖励，用于想象训练、规划与数据增强，
  将昂贵的真实交互转化为廉价的模型内 rollout。
- **RSSM（Recurrent State Space Model）**：隐状态拆成两部分——确定性态 `h_t`（GRU，
  承担记忆与时序连贯）+ 随机态 `z_t`（对角高斯，承担环境不确定性表达）。
  训练用**后验** `q(z_t|h_t,e_t)`（能看到真实帧，保证重建准确）；想象用**先验**
  `p(z_t|h_t)`（看不到真实帧，纯动力学递推）；两者用 KL 损失对齐，
  把后验学到的动力学"蒸馏"进先验——这是开环想象可用性的命脉（消融实验证实）。
- **free bits**：KL 逐元素下限（当前 0.05 nats）。没有它，训练初期 KL 会被优化到 0
  （后验≈先验但都学不到东西，"后验坍缩"），梯度消失导致先验预测糊成一团；
  加 free bits 保证最小信息流。实测 free_bits 1.0→0.2 让单步预测 +5.7 dB。
- **四损失联合训练**：`L = 重建MSE + 1.5·KL(free bits) + 2.0·奖励MSE + 1.0·价值MSE`，
  价值目标为 20 步窗口折扣回报（γ=0.99），为后续 Dreamer 式 actor-critic 打基础。
- **开环想象**：`imagine_rollout` 仅凭首帧后验 + 动作序列自回归展开，全程不接真实图像；
  误差随步长累积（1 步 27.7dB → 20 步 21.8dB），建议想象 horizon 取 10~16 步。

## 三、模型结构细节

DreamerV2 风格三段式世界模型：

```
   x_t (64×64×3)             a_t (4)
        │                      │
        ▼                      │
  ┌──────────────┐             │
  │ ConvEncoder  │  mu (64)    │       复用 AE 预训练权重
  └──────┬───────┘             │
         └──────────┬──────────┘
                    ▼
        ┌───────────────────────────┐
        │  RSSM  (每步 t)           │
        │  h_t = GRU([z_{t-1},a_t], │
        │            h_{t-1})       │
        │  后验 z_t ~ q(z|h_t,e_t)   │  ← observe：看到真实帧
        │  先验 z_t ~ p(z|h_t)       │  ← imagine_step：纯动力学预测
        └─────────────┬─────────────┘
                      ▼  feature = [h_t, z_t] (264)
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ ConvDecoder  │ │ Reward Head  │ │ Value Head   │
│ → x̂_t (图像) │ │ → r̂_t (奖励) │ │ → V̂_t (价值) │
└──────────────┘ └──────────────┘ └──────────────┘
```

- **ConvEncoder**：4 层卷积（通道 32/64/128/256）+ GroupNorm + SiLU → mu / logvar（默认 64 维）。
- **RSSM**：GRU 确定性态 `h`(200) + 对角高斯随机态 `z`(64，tanh 限幅)，拼接为 264 维特征。
- **ConvDecoder**：4 层转置卷积，隐特征 → 64×64×3（Tanh 输出 [-1,1]）。
- **Reward Head / Value Head**：264 → 256 → 128 → 1 MLP（继承 `nn.Sequential`，权重
  key 兼容旧版），分别预测单步奖励与状态价值（折扣回报监督）。

三种推理口径（评估时严格区分）：

| 口径 | 做法 | 语义 |
|---|---|---|
| 纯 AE 重建 | 图像 → encoder mu → decoder | 无动力学，单帧压缩重建基线 |
| WM 后验重建 | observe 真实 x_t → decoder | 看到真值后的重建，**上界** |
| WM 先验单步预测 | 真实前帧后验 + imagine_step(a_t) → decoder | 不看真值，**真正的预测** |

---

## 四、已验证效果（统一评估集）

评估集：覆盖全部 100 条轨迹，每条均匀取 4 个 20 步窗口，共 **400 序列 × 19 个评分帧**
（t=1..19 三种口径对齐比较）。图像归一化到 [-1,1]，PSNR 峰值取 2²=4。
完整数据见 `outputs/wm_fb02/metrics/single_step_eval.json`（输出目录不入 git）。

### 4.1 图像重建 / 预测

| 口径 | PSNR (dB) | SSIM |
|---|---|---|
| 纯 AE 重建 | 29.69 ± 2.79 | 0.9225 ± 0.0344 |
| WM 后验重建（上界） | **33.00 ± 2.75** | **0.9651 ± 0.0114** |
| WM 先验单步预测 | **29.08 ± 2.41** | **0.9474 ± 0.0163** |

- 后验重建优于纯 AE，说明 RSSM 隐状态在压缩之外保留了额外时序/结构信息。
- 先验单步预测与后验的差距缩小到约 4 dB，桌面、红点目标、机械臂轮廓清晰可辨。
- 口径说明：本评估覆盖含专家难例的全部轨迹，比早期"随机帧整体采样"口径更严格，
  因此纯 AE 的绝对 PSNR 数值与早期单次采样结果不完全可比；同表内三口径横向可比。

**关键调参证据（free bits 修正）**：初版 `free_bits=1.0` 时真实 KL 约 0.92 被全部 clamp、
KL 梯度为零，先验预测仅 23.38 dB / SSIM 0.734（画面糊）。诊断后改为 `free_bits=0.2`
并重训 4000 步，真实 KL 从 0.92 降到 0.17，先验预测提升到 29.08 dB / SSIM 0.947
（**+5.7 dB**），后验-先验差距从 9 dB 缩到 4 dB。这验证了「先验学习压力不足是模糊主因」
的诊断。诊断工具：`trainer/diag_kl.py`。

### 4.2 奖励预测（原始奖励尺度）

| 口径 | MAE | RMSE | bias |
|---|---|---|---|
| 后验奖励 | 0.1494 | 0.9658 | -0.076 |
| 先验奖励 | 0.1506 | 0.9756 | -0.077 |

样本量 7600。MAE 约 0.15（奖励量级最大 ~10），方向性可用；RMSE 偏高源于少数
"成功推到目标"的高奖励帧难以精确回归（奖励稀疏、分布长尾）。

### 4.3 多步开环预测与消融实验

详见 `docs/多步长预测全面评估报告.md` 与 `docs/消融实验与工程优化报告.md`。

**20 步开环衰减（基线 stoch64+KL）**：PSNR 27.69 → 21.75 dB（衰减 5.94），
SSIM 0.92 → 0.84；奖励预测全程 MAE < 0.007（归一化尺度）。衰减曲线
`demos/prediction/psnr_decay_base.png`。

**消融实验核心结论**（统一口径 10 条开环 × 20 步）：

| 组别 | PSNR@1 | PSNR@10 | PSNR@20 | 1→20 衰减 |
|---|---|---|---|---|
| 基线（KL, stoch64） | 27.69 | 20.97 | 21.75 | 5.94 |
| stoch32（KL） | **28.26** | **22.72** | 22.31 | 5.94 |
| stoch16（KL） | 27.17 | 22.18 | **22.33** | **4.85** |
| 无 KL（stoch64） | 13.60 | 9.70 | 9.39 | 4.21† |

> † 无 KL 组整体崩塌（SSIM≈0.15→0.005，奖励误差 ×12），衰减小是起点就低的假象。
> 维度存在最优点：stoch32 全程精度最高、stoch16 长程最稳，均优于 stoch64；
> **KL 是开环想象的命脉，不可移除**。图表：`demos/ablation/psnr_by_dim.png`、
> `psnr_by_kl.png`、`ssim_compare.png`、`stability_bar.png`。

### 4.4 演示视频与可视化（已入库，可直接查看）

| 素材 | 内容 |
|---|---|
| `demos/prediction/real_vs_pred20_traj0002.gif` | **核心演示**：真实 vs 20 步开环预测并排动图（仅首帧+动作，全程不看真实帧） |
| `demos/prediction/real_vs_pred20_traj0002_grid.png` | 上排真实 / 下排预测静态网格 |
| `demos/prediction/infer_traj0002_pred.gif` | 一键推理脚本产物：输入动作序列直接生成预测视频 |
| `demos/prediction/psnr_decay_base.png` | 20 步 PSNR/SSIM 衰减曲线 |
| `demos/ablation/psnr_by_dim.png` | 维度消融 PSNR 曲线（stoch32 全程最优） |
| `demos/ablation/psnr_by_kl.png` | KL 消融曲线（无 KL 从第 1 步即崩塌） |
| `demos/ablation/stability_bar.png` | 长程稳定性柱状图 |
| `demos/evaluation/training_curves_full.png` | 全量训练监控：三条核心损失 + 周期性 SSIM@20 达标曲线 |

单步口径 GIF（`outputs/wm_fb02/vis/`）不入 git，可用 `trainer/make_prediction_gif.py` 重新生成。

### 4.5 数据扩充与全量调参成果（详情见 `docs/数据扩充与全量调参报告.md`）

混合策略采集 + 标准化格式化后，训练集扩至 200 条 / 8324 帧（原 100 条 / 3324 帧），
立方体初始位置跨度从 0.06m 扩大到 0.23m。采用分组学习率 + 序列长度课程 + 轻量增强
全量训练 5000 步（约 8 分钟），周期性开环评估 SSIM 稳步提升：

| 训练步数 | 1000 | 2000 | 3000 | 4000 | 5000 |
|---|---|---|---|---|---|
| 20 步开环 SSIM | 0.560 | 0.668 | 0.706 | 0.711 | **0.725** |

测试集 15 条轨迹独立验证：**20 步 SSIM=0.7511（目标≥0.6）**、奖励 MAE=0.024（目标<0.5），
两项达标目标均大幅通过。重建损失 0.33→0.0093，KL 稳定 0.05 无坍缩。

### 4.6 代码结构化与消融可视化成果（详情见 `docs/消融与可视化报告.md`）

**模块解耦**：世界模型从单一 `vae.py` 拆为 `encoder.py` / `decoder.py` / `rssm.py` /
`reward_predictor.py` 独立模块，新增 `factory.py` 模型工厂统一构建；`vae.py` / `heads.py`
保留为兼容转发层，历史检查点权重 key 完全不变（已用 `wm_best.pt` 回归验证无缝加载）。
超参数由 `configs/default.yaml` 单一来源管理（含新增 `model.dynamics`），消除脚本硬编码。

**预测对比可视化**（`results/visualization/`，3 条测试轨迹）：真实帧 vs 20 步开环预测并排视频，
逐帧标注 PSNR/SSIM，5/10/20 步关键节点高亮，并附三联对比图与衰减曲线。示例轨迹：
PSNR 从 1 步约 26dB 衰减到 20 步 19~23dB，直观体现长程误差累积规律。

**统一消融实验**（同一扩充数据 + 同一超参 + 均不开增强，`results/`，详情见 `docs/消融与可视化报告.md`）：
- 维度消融（stoch 16/32/48/64）：20 步 PSNR 23.14/23.75/23.71/24.00 dB，
  维度升高精度略好但边际递减（16→64 仅 +0.86dB），32 维长程衰减最小（5.81dB）；
- 动力学消融：RSSM 20 步 PSNR 23.71dB / SSIM 0.869，纯 MLP 基线仅 17.61dB / 0.398，
  证明循环确定性态 + 随机隐状态是长程预测能力的关键；
- 统一评估口径 1/5/10/20 步 PSNR/SSIM/奖励误差，图表与指标表入库 `results/`。

---

## 五、训练 / 评估启动方式

全部命令在 `wzh-ai` 容器的 `vla` conda 环境执行（解释器：`/opt/miniconda3/envs/vla/bin/python`，Python 3.12 + torch 2.6.0+cu124，GPU：NVIDIA RTX A6000）。

### 5.0 快速启动（最小流程，约 15 分钟跑通全链路）

```bash
cd /data/wzh/robot/vla-world-model-lab
VLA=/opt/miniconda3/envs/vla/bin/python

# ① 训练视觉自编码器（约 1 分钟，提供 encoder/decoder 初始权重）
$VLA trainer/train_ae.py --config configs/world_model.yaml --max-steps 1000

# ② 世界模型四损失联合训练（GPU 约 7 分钟）
$VLA trainer/train_world_model.py --config configs/default.yaml --output-dir outputs/wm

# ③ 20 步开环预测全面评估（PSNR 衰减 + 曲线 + 对比视频）
$VLA trainer/eval_multistep_full.py --ckpt outputs/wm/checkpoints/wm_step004000.pt --out-dir outputs/wm

# ④ 一键推理：输入动作序列，直接生成预测视频
$VLA trainer/infer_video.py --ckpt outputs/wm/checkpoints/wm_step004000.pt     --traj traj_0002 --start 8 --steps 20 --out outputs/infer_demo

# ⑤ 查看结果：outputs/wm/vis/*.gif、psnr_decay.png、outputs/infer_demo/*
```

> 前置依赖：`outputs/data/trajectories.h5`（100 条轨迹）与 `outputs/checkpoints/ae_best.pt`。
> 数据由 `mujoco-studio-lab` 仓库的 `scripts/export_trajectories.py` 导出（MuJoCo EGL 离屏渲染）。

### 5.1 完整实验命令流水线（复现全部实验）


```bash
cd /data/wzh/robot/vla-world-model-lab
VLA=/opt/miniconda3/envs/vla/bin/python

# 1) 数据自检（可选，确认数据集可用）
$VLA datasets/episodic_dataset.py --batch-size 8

# 2) 视觉自编码器训练（GPU 预加载加速，约 51s / 1000 步）
$VLA trainer/train_ae.py --config configs/world_model.yaml --max-steps 1000

# 3) AE 效果验收（PSNR + 对比图）
$VLA trainer/eval_ae.py --ckpt outputs/checkpoints/ae_best.pt

# 4) 世界模型三损失联合训练（约 395s / 4000 步，free_bits=0.2）
$VLA trainer/train_world_model.py --config configs/default.yaml --output-dir outputs/wm_fb02

# 5) KL 诊断（确认先验学到：真实 KL 应从 0.92 降到约 0.17）
$VLA trainer/diag_kl.py outputs/wm_fb02/checkpoints/wm_step004000.pt

# 6) 单步预测全面验证（PSNR / SSIM / 奖励误差 → JSON）
$VLA trainer/eval_single_step.py --ckpt outputs/wm_fb02/checkpoints/wm_step004000.pt --out-dir outputs/wm_fb02

# 7) 连续 10 步预测对比 GIF
$VLA trainer/make_prediction_gif.py --ckpt outputs/wm_fb02/checkpoints/wm_step004000.pt --out-dir outputs/wm_fb02/vis

# 8) 开环多步想象预测（仅首帧+动作，输出对比动图）
$VLA trainer/test_open_loop.py --ckpt outputs/wm_fb02/checkpoints/wm_step004000.pt --steps 10

# 9) 奖励头 + 价值头四损失联合训练（约 433s / 4000 步）
$VLA trainer/train_world_model.py --config configs/default.yaml --output-dir outputs/wm_heads

# 10) 奖励/价值曲线验证（corr + MAE + 对比图 → JSON）
$VLA trainer/verify_heads.py --ckpt outputs/wm_heads/checkpoints/wm_step004000.pt

# 11) 最优模型（KL调优，配置已固化）：四损失训练
$VLA trainer/train_world_model.py --config configs/default.yaml --output-dir outputs/wm_tune1

# 12) 多轨迹10步开环基准评估（逐 step PSNR/SSIM）
$VLA trainer/eval_multistep.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt --out-dir outputs/wm_tune1

# 13) 立方体聚焦可视化（第10步立方体是否可辨）
$VLA trainer/vis_cube_focus.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt --out outputs/wm_tune1/vis/cube_focus.gif

# 14) 多步长全面评估（1/5/10/20 步 PSNR 衰减 + 奖励误差 + 折线图 + 3组轨迹视频）
$VLA trainer/eval_multistep_full.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt

# 15) 想象策略训练最小 Demo（固定起点，想象回报单调上升 + 趋势图）
$VLA trainer/imagine_policy.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt --iters 300 --fixed-init

# 16) 想象策略训练（价值头 baseline 版，Dreamer critic 降方差）
$VLA trainer/imagine_policy.py --iters 300 --fixed-init --baseline

# 17) 混合策略数据扩充（需 mujoco 环境，EGL 离屏渲染，约 13 分钟）
/opt/miniconda3/envs/mujoco/bin/python scripts/collect_mixed_trajectories.py \
    --out outputs/data/mixed_raw.h5 --n-ppo 50 --n-random 50 --max-steps 50

# 18) 标准化格式化 + 9:1 划分 + 数据质量自检
$VLA scripts/format_dataset.py --raw outputs/data/mixed_raw.h5 \
    --extra outputs/data/trajectories.h5 --out-dir outputs/data/wm_dataset
$VLA scripts/verify_dataset.py

# 19) 全量训练（分组学习率 + 课程 + 增强 + 周期性评估存最优，约 8 分钟）
bash scripts/train_full.sh

# 20) 达标验证 + 训练曲线图
$VLA scripts/verify_targets.py --ckpt outputs/wm_mixtrain/checkpoints/wm_best.pt
$VLA scripts/plot_training_curves.py

# 21) 多步预测对比可视化（3 条测试轨迹 → 并排视频 + 5/10/20 步关键节点图）
$VLA scripts/compare_video.py --out-dir results/visualization

# 22) 统一消融训练（维度 16/32/64 + MLP 基线；基线 stoch48 见步骤 19）
bash scripts/train_ablation_unified.sh

# 23) 统一消融评估 + 汇总图表（改 results/abl_groups.json 指向各组 wm_best.pt）
$VLA scripts/eval_ablation_unified.py
$VLA scripts/plot_ablation_unified.py
```

`--config` 只填充命令行未显式给出的参数，显式参数始终优先，例如：

```bash
$VLA trainer/train_world_model.py --config configs/default.yaml --max-steps 5000
```

数据若需重新导出，用 `mujoco` 环境执行 `scripts/export_trajectories.py`（需 EGL 离屏渲染）。

---

## 六、已知限制与下一步

1. **先验与后验仍有约 4 dB 差距**：free bits 从 1.0 降到 0.2 后先验预测从 23.38 提升到
   29.08 dB（已解决"糊"的主要问题），但纯动力学预测仍比看到真值的后验重建略模糊。
   下一步：进一步调低 free bits（0.05~0.1）、提高 KL scale，或加入多步想象损失。
2. **离散隐变量未采用**：当前用对角高斯随机态（tanh 限幅）。下一步可尝试 DreamerV2 的
   categorical 隐变量以提升多步预测稳定性。
3. **奖励尖峰幅值低估**：奖励头与真实曲线相关系数 0.767（趋势一致），但成功帧尖峰
   （≈1.0）只预测到约 0.05——极稀疏奖励（>0.5 的帧仅 2.56%）下 MSE 倾向预测条件期望。
4. **价值预测在专家轨迹末段偏低**：相关系数 0.55、MAE 1.06。随机 20 步窗口折扣回报
   均值近 0，价值头学到"预测接近 0"；专家轨迹末段累积高奖励属分布外外推。
5. **想象策略 Demo 存在 model exploitation**：想象回报训练中 1.32→2.59 单调上升（链路已验证），
   但专家动作的想象回报(0.48)反而低于随机策略(1.69)——奖励头在后验特征上训练、想象用先验特征
   有分布偏移，策略学会了利用奖励头偏好。**"想象回报上升"不等于"真实性能上升"**。下一步需
   λ-return + 价值引导 + 策略蒸馏 + 真实环境定期评估。详见 `docs/想象策略训练Demo说明.md`。
6. 已做轨迹级 9:1 held-out 划分（train/test 无重叠），但测试轨迹与训练轨迹同分布（同一任务/场景），
   指标反映同分布重建 / 预测保真度，跨场景（新目标位置/新光照）泛化能力未单独验证。

---

## 七、目录结构

```
vla-world-model-lab/
├── configs/         # default.yaml（推荐）+ world_model.yaml（早期兼容）
├── datasets/        # episodic_dataset.py（序列数据集）、fast_loader.py（GPU 预加载）
├── models/          # vae.py（编码器/解码器）、rssm.py（RSSM）、heads.py（奖励/价值头）、policy.py（策略网络）
├── trainer/         # 训练与评估脚本（train_ae / train_world_model / eval / verify / gif）
├── utils/           # 工具模块
├── outputs/         # 数据、检查点、日志、可视化、指标（不入 git）
├── scripts/         # 数据导出脚本
├── demos/           # 演示与结果
├── docs/            # 中文技术文档
└── README.md
```

## 关联仓库

- `mujoco-studio-lab`：PushCube 环境与专家轨迹来源
- `embodied-policy-zoo`：策略算法库（BC / ACT / PPO / IQL / DT），世界模型后续可为其提供想象数据
