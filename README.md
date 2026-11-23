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
| 四损失联合训练 `trainer/train_world_model.py` | ✅ | 重建 + KL(free bits) + 奖励 + 价值，4000 步收敛 |
| 单步预测全面验证 `trainer/eval_single_step.py` | ✅ | PSNR / SSIM / 奖励 MAE·RMSE，统一评估集 |
| 预测可视化 `trainer/make_prediction_gif.py` | ✅ | 连续 10 步真实 vs 单步预测对比 GIF |
| 开环多步预测 `models/rssm.py` `imagine_rollout` | ✅ | 仅首帧+动作，全程不接真实图像，任意步长 |
| 奖励头 / 价值头 `models/heads.py` | ✅ | 奖励单步回归 + 状态价值（折扣回报监督） |
| 奖励/价值曲线验证 `trainer/verify_heads.py` | ✅ | 奖励 corr 0.77，价值 corr 0.55（含局限说明） |
| 统一配置 `configs/default.yaml` | ✅ | data / model / training / eval 四段，命令行优先 |
| 想象轨迹上的 actor-critic 规划 | ⬜ | 下一阶段目标 |

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

## 三、已验证效果（统一评估集）

评估集：覆盖全部 100 条轨迹，每条均匀取 4 个 20 步窗口，共 **400 序列 × 19 个评分帧**
（t=1..19 三种口径对齐比较）。图像归一化到 [-1,1]，PSNR 峰值取 2²=4。
完整数据见 `outputs/wm_fb02/metrics/single_step_eval.json`（输出目录不入 git）。

### 3.1 图像重建 / 预测

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

### 3.2 奖励预测（原始奖励尺度）

| 口径 | MAE | RMSE | bias |
|---|---|---|---|
| 后验奖励 | 0.1494 | 0.9658 | -0.076 |
| 先验奖励 | 0.1506 | 0.9756 | -0.077 |

样本量 7600。MAE 约 0.15（奖励量级最大 ~10），方向性可用；RMSE 偏高源于少数
"成功推到目标"的高奖励帧难以精确回归（奖励稀疏、分布长尾）。

### 3.3 可视化

- `prediction_single_step.gif`：连续 10 步真实帧 vs 单步预测帧逐帧对比动图。
- `prediction_single_step.png`：2×10 静态对比网格（上排真实、下排预测）。

（以上位于 `outputs/wm_fb02/vis/`，输出目录不入 git，可运行脚本重新生成。）

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
```

`--config` 只填充命令行未显式给出的参数，显式参数始终优先，例如：

```bash
$VLA trainer/train_world_model.py --config configs/default.yaml --max-steps 5000
```

数据若需重新导出，用 `mujoco` 环境执行 `scripts/export_trajectories.py`（需 EGL 离屏渲染）。

---

## 五、已知限制与下一步

1. **先验与后验仍有约 4 dB 差距**：free bits 从 1.0 降到 0.2 后先验预测从 23.38 提升到
   29.08 dB（已解决"糊"的主要问题），但纯动力学预测仍比看到真值的后验重建略模糊。
   下一步：进一步调低 free bits（0.05~0.1）、提高 KL scale，或加入多步想象损失。
2. **离散隐变量未采用**：当前用对角高斯随机态（tanh 限幅）。下一步可尝试 DreamerV2 的
   categorical 隐变量以提升多步预测稳定性。
3. **奖励尖峰幅值低估**：奖励头与真实曲线相关系数 0.767（趋势一致），但成功帧尖峰
   （≈1.0）只预测到约 0.05——极稀疏奖励（>0.5 的帧仅 2.56%）下 MSE 倾向预测条件期望。
4. **价值预测在专家轨迹末段偏低**：相关系数 0.55、MAE 1.06。随机 20 步窗口折扣回报
   均值近 0，价值头学到"预测接近 0"；专家轨迹末段累积高奖励属分布外外推。
5. **尚未接入 actor-critic**：奖励/价值头已在真实轨迹上训练，但还没在想象 rollout 上
   计算 λ-return 与优势，未用于规划。这是下一步目标。
6. 当前为全量训练、未做轨迹级 held-out 划分，指标反映重建 / 预测保真度，非严格泛化能力。

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
