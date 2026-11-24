#!/bin/bash
# 消融实验训练编排：3 组新模型 + 复用基线
# 在 vla-world-model-lab 根目录运行
set -e
cd /data/wzh/robot/vla-world-model-lab
P=/opt/miniconda3/envs/vla/bin/python

echo "========== 消融 A: 隐状态维度 stoch_dim = 16 =========="
$P trainer/train_world_model.py --config configs/default.yaml \
  --stoch-dim 16 --max-steps 4000 --output-dir outputs/abl_stoch16 \
  2>&1 | tee outputs/abl_stoch16_train.log

echo "========== 消融 A: 隐状态维度 stoch_dim = 32 =========="
$P trainer/train_world_model.py --config configs/default.yaml \
  --stoch-dim 32 --max-steps 4000 --output-dir outputs/abl_stoch32 \
  2>&1 | tee outputs/abl_stoch32_train.log

echo "========== 消融 B: 无 KL 损失 (kl_scale=0) stoch_dim=64 =========="
$P trainer/train_world_model.py --config configs/default.yaml \
  --stoch-dim 64 --kl-scale 0 --max-steps 4000 --output-dir outputs/abl_nokl \
  2>&1 | tee outputs/abl_nokl_train.log

echo "========== 全部训练完成 =========="
