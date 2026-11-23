# vla-world-model-lab

具身智能世界模型实验室：从离线交互轨迹学习「图像 → 隐状态 → 图像」的动力学模型，
为后续 VLA 策略的想象 rollout / 数据增强提供基础。**全程在 `vla` conda 环境开发。**

## 当前进度

- [x] 世界模型基础依赖（torchvision / einops / opencv / matplotlib / tensorboard / h5py）
- [x] 离线轨迹数据集导出（约 100 条，40 专家 + 60 随机，HDF5）
- [x] 序列数据集类 `datasets/episodic_dataset.py`（20 步序列 / 64×64 / 归一化）
- [x] DreamerV2 核心理论笔记
- [x] 视觉自编码器（编码器 + 解码器，64 维隐变量；MSE 0.006 / PSNR 41dB，全量训练 51 秒）
- [x] 自编码器训练脚本（MSE / LR 衰减 / 检查点 / TensorBoard / 重建可视化；GPU 内存预加载 80x 加速）
- [x] RSSM 动力学骨架（observe / imagine_step / get_feature，特征 264 维）
- [ ] 奖励头 + 单步预测训练
- [ ] 单步预测训练与重建可视化

## 目录结构

```
vla-world-model-lab/
├── models/          # 编码器、解码器、RSSM、奖励头模型定义
├── trainer/         # 训练循环、损失函数、评估逻辑
├── datasets/        # 数据集加载与预处理（episodic_dataset.py）
├── configs/         # 超参数配置（world_model.yaml）
├── utils/           # 可视化、指标计算工具
├── outputs/         # 训练日志、检查点、可视化结果（不入 git）
│   └── data/        # trajectories.h5（离线轨迹数据）
├── scripts/         # 数据导出 / 训练 / 评估脚本
├── demos/           # 演示视频与结果
└── docs/            # 中文文档（DreamerV2 理论笔记等）
```

## 快速开始

```bash
# 数据与数据集自检（vla 环境）
/opt/miniconda3/envs/vla/bin/python datasets/episodic_dataset.py --batch-size 8

# 重新导出离线轨迹（mujoco 环境，需 EGL 离屏渲染）
cd /data/wzh/robot/mujoco-studio-lab
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless \
  /opt/miniconda3/envs/mujoco/bin/python /data/wzh/robot/vla-world-model-lab/scripts/export_trajectories.py
```

## 关联仓库

- `mujoco-studio-lab`：PushCube 环境与专家轨迹来源
- `embodied-policy-zoo`：策略算法库（BC/ACT/PPO/IQL/DT），世界模型后续可为其提供想象数据
