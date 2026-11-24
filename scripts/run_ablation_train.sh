#!/bin/bash
# 消融实验 消融训练：步长(5/10/20) + 奖励来源(wm/zero)。vla 环境 GPU。
# 固定满数据世界模型 abl2_stoch48，只变单一因素。
set -u
cd /data/wzh/robot/vla-world-model-lab
PY=/opt/miniconda3/envs/vla/bin/python
CKPT=outputs/abl2_stoch48/checkpoints/wm_best.pt
DATA=outputs/data/trajectories.h5
MASTER=outputs/ablation_train_master.log
echo "========== 消融实验 消融训练开始 $(date) ==========" > "$MASTER"

train_one () {
  local TAG=$1 H=$2 REWARD=$3
  local OUT="outputs/abl_${TAG}"
  echo "----- TRAIN ${TAG}: H=${H} reward=${REWARD} -> ${OUT} $(date) -----" | tee -a "$MASTER"
  OMP_NUM_THREADS=1 $PY scripts/train_imagine_ablation.py \
    --ckpt "$CKPT" --data "$DATA" \
    --horizon "$H" --reward-mode "$REWARD" \
    --epochs 400 --batch-size 128 --entropy-coef 2e-2 --lr-actor 1.5e-4 \
    --log-interval 50 --eval-interval 10 \
    --out-dir "$OUT" > "outputs/ablation_train_${TAG}.log" 2>&1
  echo "----- TRAIN ${TAG} 退出码=$? $(date) -----" | tee -a "$MASTER"
}

# 消融1：想象步长（reward=wm 固定）
train_one h5  5  wm
train_one h10 10 wm
train_one h20 20 wm
# 消融2：奖励来源（H=10 固定，zero 组；wm 组复用 h10）
train_one norew 10 zero

echo "========== 消融实验 消融训练完成 $(date) ==========" | tee -a "$MASTER"
