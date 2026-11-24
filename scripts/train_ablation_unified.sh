#!/bin/bash
# 统一口径消融训练：同一扩充数据集 + 同一超参，仅改变隐状态维度 / 动力学类型
# 基线 stoch48 已有（outputs/wm_mixtrain）。本脚本训练 4 组对照。
set -e
cd /data/wzh/robot/vla-world-model-lab
P=/opt/miniconda3/envs/vla/bin/python
DATA=outputs/data/wm_dataset/train.h5
EVAL=outputs/data/wm_dataset/test.h5
COMMON="--config configs/default.yaml --data $DATA --eval-data $EVAL --batch-size 16 \
  --group-lr --lr-enc 1e-4 --lr-dyn 3e-4 --free-bits 0.05 --kl-scale 1.5 \
  --seq-len 20 --seq-len-start 10 --curriculum-steps 30 \
  --max-steps 5000 --warmup-steps 50 --save-interval 2500 \
  --eval-interval 500 --eval-trajs 6 --log-interval 250 --vis-interval 5000"

echo "========== 消融: 隐状态维度 stoch_dim = 16 =========="
$P trainer/train_world_model.py $COMMON --stoch-dim 16 --output-dir outputs/abl2_stoch16 \
  2>&1 | tee outputs/abl2_stoch16_train.log

echo "========== 消融: 隐状态维度 stoch_dim = 32 =========="
$P trainer/train_world_model.py $COMMON --stoch-dim 32 --output-dir outputs/abl2_stoch32 \
  2>&1 | tee outputs/abl2_stoch32_train.log

echo "========== 消融: 隐状态维度 stoch_dim = 64 =========="
$P trainer/train_world_model.py $COMMON --stoch-dim 64 --output-dir outputs/abl2_stoch64 \
  2>&1 | tee outputs/abl2_stoch64_train.log

echo "========== 消融: 纯 MLP 动力学 (无循环/无随机) stoch_dim = 48 =========="
$P trainer/train_world_model.py $COMMON --dynamics mlp --kl-scale 0 --stoch-dim 48 \
  --output-dir outputs/abl2_mlp 2>&1 | tee outputs/abl2_mlp_train.log

echo "========== 全部消融训练完成 =========="
