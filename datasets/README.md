# 数据集说明

## 数据来源与格式

`outputs/data/trajectories.h5`（由 `scripts/export_trajectories.py` 生成，约 100 条轨迹）：

| 来源 | 条数 | 说明 |
|---|---|---|
| 专家轨迹 | 40 | 来自 mujoco-studio-lab 的 `raw_demos_v2.npz`（成功轨迹，精选） |
| 随机策略轨迹 | 60 | PushCubeEnv 随机动作采样，覆盖状态空间，不要求任务成功 |

每条轨迹组 `traj_XXXX` 含：

- `images`: (T, 224, 224, 3) uint8
- `actions`: (T, 4) float32，原始动作 `[dx,dy,dz,gripper]`
- `states`: (T, 13) float32，关节角+方块/目标位置
- `rewards`: (T,) float32，专家轨迹为按状态近似的奖励，随机轨迹为环境真实奖励
- `dones`: (T,) bool

## 序列采样（`episodic_dataset.py`）

- 单条序列固定 20 步，只在单条轨迹内滑动截取，不跨回合；
- 图像双线性下采样 64×64，归一化到 [-1, 1]；
- 动作按动作空间 `[lo,hi]` 归一化到 [-1, 1]（平移 ±0.02m，gripper 0~1）；
- 奖励除以 10 缩放到约 [-1, 1]；done 转 float32。

## 自检命令

```bash
/opt/miniconda3/envs/vla/bin/python datasets/episodic_dataset.py --batch-size 8
```
