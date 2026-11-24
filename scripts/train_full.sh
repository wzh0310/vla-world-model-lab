#!/bin/bash
# 世界模型全量训练与调参（wzh-ai 容器 vla 环境）
# 调参要点：分组学习率(enc 1e-4 / dyn 3e-4) + 课程序列长度(10→20) +
#           轻量图像增强(平移+亮度+裁剪) + 每 1000 步开环评估存最优权重
set -e
cd /data/wzh/robot/vla-world-model-lab
PY=/opt/miniconda3/envs/vla/bin/python

echo "=== 全量训练 $(date) ==="
$PY trainer/train_world_model.py \
  --config configs/default.yaml \
  --data outputs/data/wm_dataset/train.h5 \
  --eval-data outputs/data/wm_dataset/test.h5 \
  --latent-dim 64 \
  --stoch-dim 48 \
  --det-dim 200 \
  --image-size 64 \
  --seq-len 20 \
  --seq-len-start 10 \
  --curriculum-steps 1000 \
  --batch-size 32 \
  --group-lr \
  --lr-enc 1e-4 \
  --lr-dyn 3e-4 \
  --free-bits 0.05 \
  --kl-scale 1.5 \
  --kl-warmup 0 \
  --reward-scale 2.0 \
  --value-scale 1.0 \
  --augment \
  --shift-px 2 \
  --brightness 0.1 \
  --crop-frac 0.08 \
  --max-steps 5000 \
  --warmup-steps 100 \
  --eval-interval 1000 \
  --eval-trajs 6 \
  --save-interval 1000 \
  --log-interval 100 \
  --vis-interval 1000 \
  --seed 0 \
  --output-dir outputs/wm_mixtrain
echo "=== 训练完成 $(date) ==="
