# vla-world-model-lab

具身智能世界模型实验室：从离线交互轨迹学习「图像 + 动作 → 隐状态 → 下一帧图像」的动力学模型，
为后续 VLA 策略的想象 rollout / 数据增强 / 规划提供基础。**全程在 `vla` conda 环境开发。**

当前阶段：**单步预测世界模型可用** —— 已完成视觉自编码器、RSSM 动力学、三损失联合训练，
并在统一评估集上完成 PSNR / SSIM / 奖励误差的量化验证与连续 10 步预测可视化。

---

## 一、当前进度

| 模块 | 状态 | 说明 |
|---|---|---|
| 离线轨迹数据底座 | ✅ | 100 条轨迹（40 专家 + 60 随机），HDF5，约 70MB，不入 git |
| 序列数据集 `datasets/episodic_dataset.py` | ✅ | 20 步序列 / 64×64 / 动作与奖励归一化 |
| GPU 显存预加载采样 `datasets/fast_loader.py` | ✅ | 约 80× 加速（1000 步从 ~70min → 51s） |
| 视觉自编码器 `models/vae.py` | ✅ | ConvEncoder/ConvDecoder，64 维隐变量 |
| RSSM 动力学 `models/rssm.py` | ✅ | 确定性态 200 + 随机态 64，特征 264 维 |
| 三损失联合训练 `trainer/train_world_model.py` | ✅ | 重建 + KL(free bits) + 奖励，3000 步收敛 |
| 单步预测全面验证 `trainer/eval_single_step.py` | ✅ | PSNR / SSIM / 奖励 MAE·RMSE，统一评估集 |
| 预测可视化 `trainer/make_prediction_gif.py` | ✅ | 连续 10 步真实 vs 单步预测对比 GIF |
| 统一配置 `configs/default.yaml` | ✅ | data / model / training / eval 四段，命令行优先 |
| 多步想象 rollout + actor-critic 规划 | ⬜ | 下一阶段目标 |

---

## 二、模型架构

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
              ┌───────┴────────┐
              ▼                ▼
      ┌──────────────┐  ┌──────────────┐
      │ ConvDecoder  │  │ Reward Head  │
      │ → x̂_t (图像) │  │ → r̂_t (标量) │
      └──────────────┘  └──────────────┘
```

- **ConvEncoder**：4 层卷积（通道 32/64/128/256）+ GroupNorm + SiLU → mu / logvar（默认 64 维）。
- **RSSM**：GRU 确定性态 `h`(200) + 对角高斯随机态 `z`(64，tanh 限幅)，拼接为 264 维特征。
- **ConvDecoder**：4 层转置卷积，隐特征 → 64×64×3（Tanh 输出 [-1,1]）。
- **Reward Head**：264 → 256 → 128 → 1 MLP，预测奖励。

三种推理口径（评估时严格区分）：

| 口径 | 做法 | 语义 |
|---|---|---|
| 纯 AE 重建 | 图像 → encoder mu → decoder | 无动力学，单帧压缩重建基线 |
| WM 后验重建 | observe 真实 x_t → decoder | 看到真值后的重建，**上界** |
| WM 先验单步预测 | 真实前帧后验 + imagine_step(a_t) → decoder | 不看真值，**真正的预测** |

---

## 三、已验证效果（统一评估集）

评估集：覆盖全部 100 条轨迹，每条均匀取 4 个 20 步窗口，共 **400 序列 × 19 个评分帧**
（t=1..19 三种口径对齐比较）。图像归一化到 [-1,1]，PSNR 峰值取 2²=4。
完整数据见 `outputs/wm/metrics/single_step_eval.json`（输出目录不入 git）。

### 3.1 图像重建 / 预测

| 口径 | PSNR (dB) | SSIM |
|---|---|---|
| 纯 AE 重建 | 29.69 ± 2.79 | 0.9225 ± 0.0344 |
| WM 后验重建（上界） | **32.57 ± 2.83** | **0.9602 ± 0.0144** |
| WM 先验单步预测 | 23.38 ± 1.80 | 0.7343 ± 0.0802 |

- 后验重建优于纯 AE，说明 RSSM 隐状态在压缩之外保留了额外时序/结构信息。
- 先验单步预测比后验低约 9 dB，但仍能辨认桌面、红点目标与机械臂轮廓；
  这是 VAE 类世界模型纯动力学预测的典型现象（详见「已知限制」）。
- 口径说明：本评估覆盖含专家难例的全部轨迹，比早期"随机帧整体采样"口径更严格，
  因此纯 AE 的绝对 PSNR 数值与早期单次采样结果不完全可比；同表内三口径横向可比。

### 3.2 奖励预测（原始奖励尺度）

| 口径 | MAE | RMSE | bias |
|---|---|---|---|
| 后验奖励 | 0.1493 | 0.9661 | -0.076 |
| 先验奖励 | 0.1484 | 0.9848 | -0.086 |

样本量 7600。MAE 约 0.15（奖励量级最大 ~10），方向性可用；RMSE 偏高源于少数
"成功推到目标"的高奖励帧难以精确回归（奖励稀疏、分布长尾）。

### 3.3 可视化

- `prediction_single_step.gif`：连续 10 步真实帧 vs 单步预测帧逐帧对比动图。
- `prediction_single_step.png`：2×10 静态对比网格（上排真实、下排预测）。

（以上位于 `outputs/wm/vis/`，输出目录不入 git，可运行脚本重新生成。）

---

## 四、训练 / 评估启动方式

全部命令在容器 `wzh-ai` 的 `vla` 环境执行（解释器：`/opt/miniconda3/envs/vla/bin/python`）。

```bash
cd /data/wzh/robot/vla-world-model-lab
VLA=/opt/miniconda3/envs/vla/bin/python

# 1) 数据自检（可选，确认数据集可用）
$VLA datasets/episodic_dataset.py --batch-size 8

# 2) 视觉自编码器训练（GPU 预加载加速，约 51s / 1000 步）
$VLA trainer/train_ae.py --config configs/world_model.yaml --max-steps 1000

# 3) AE 效果验收（PSNR + 对比图）
$VLA trainer/eval_ae.py --ckpt outputs/checkpoints/ae_best.pt

# 4) 世界模型三损失联合训练（约 326s / 3000 步）
$VLA trainer/train_world_model.py --config configs/default.yaml

# 5) 单步预测全面验证（PSNR / SSIM / 奖励误差 → JSON）
$VLA trainer/eval_single_step.py --ckpt outputs/wm/checkpoints/wm_step003000.pt

# 6) 连续 10 步预测对比 GIF
$VLA trainer/make_prediction_gif.py --ckpt outputs/wm/checkpoints/wm_step003000.pt
```

`--config` 只填充命令行未显式给出的参数，显式参数始终优先，例如：

```bash
$VLA trainer/train_world_model.py --config configs/default.yaml --max-steps 5000
```

数据若需重新导出，用 `mujoco` 环境执行 `scripts/export_trajectories.py`（需 EGL 离屏渲染）。

---

## 五、已知限制与下一步

1. **KL 恒为阈值 1.0**：free bits 下限使 KL 项几乎不产生梯度，先验学习压力不足，
   是单步预测偏模糊的主因。下一步：调低 free bits（0.1~0.3）或加入多步想象损失。
2. **先验预测偏模糊**：23 dB / SSIM 0.73，结构在但纹理缺失。下一步：更长训练、提高 KL scale，
   或改用 DreamerV2 的离散（categorical）隐变量。
3. **奖励长尾帧误差**：高奖励稀疏帧 RMSE 偏高。下一步：加权损失或分桶评估。
4. **尚未接入 actor-critic**：世界模型目前仅用于重建 / 预测验证，未用于规划或想象 rollout。
5. 当前为全量训练、未做轨迹级 held-out 划分，指标反映重建 / 预测保真度，非严格泛化能力。

---

## 六、目录结构

```
vla-world-model-lab/
├── configs/         # default.yaml（推荐）+ world_model.yaml（早期兼容）
├── datasets/        # episodic_dataset.py（序列数据集）、fast_loader.py（GPU 预加载）
├── models/          # vae.py（编码器/解码器）、rssm.py（RSSM 动力学）
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
