# 模型权重

本目录用于存放训练产出的检查点。**权重文件不纳入 Git**（体积约 40MB / 个，见本目录 `.gitignore`），
请通过下列两种方式之一获取：

## 方式一：按 README 从零训练（推荐，可完整复现）

```bash
VLA=/opt/miniconda3/envs/vla/bin/python

# 1) 视觉自编码器（提供 encoder / decoder 初始权重）
$VLA trainer/train_ae.py --config configs/world_model.yaml --max-steps 1000
#    -> outputs/checkpoints/ae_best.pt

# 2) 世界模型四损失联合训练
bash scripts/train_full.sh
#    -> outputs/wm_mixtrain/checkpoints/wm_best.pt  （项目主用检查点）
```

## 方式二：把已有检查点放到本目录

将训练好的 `.pt` 文件复制到 `weights/` 后，用 `--ckpt` 显式指定：

```bash
$VLA scripts/eval_world_model.py --ckpt weights/wm_best.pt --out-dir outputs/eval_local
$VLA scripts/visualize_predictions.py --ckpt weights/wm_best.pt
```

## 检查点内容约定

| 文件 | 内容 | 用途 |
|---|---|---|
| `ae_best.pt` | ConvEncoder / ConvDecoder | 自编码器预训练权重 |
| `wm_best.pt` | Encoder + RSSM + Decoder + Reward/Value Head | 世界模型主检查点 |
| `wm_step*.pt` | 训练中途快照 | 曲线 / 课程学习复盘 |
| `imagine_policy.pt` | Actor / Critic | 隐空间想象策略 |

> 权重 key 在模块拆分（vae.py → encoder.py / decoder.py 等）前后完全兼容，
> 历史检查点可直接用新代码加载；兼容转发层见 `models/vae.py`、`models/heads.py`、`trainer/metrics.py`。
