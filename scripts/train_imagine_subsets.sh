#!/bin/bash
# 世界模型样本效率实验：为每个数据比例的世界模型训练想象策略。
# 沿用 v2 已验证超参（epochs400/batch128/H12/熵2e-2），唯一变量为 WM checkpoint。
set -u
cd /data/wzh/robot/vla-world-model-lab
PY=/opt/miniconda3/envs/vla/bin/python
MASTER=outputs/sfe_imagine_master.log
echo "========== 想象策略子集训练开始 $(date) ==========" > "$MASTER"

for PCT in 10 30 50 100; do
  CKPT="outputs/sfe_wm_${PCT}/checkpoints/wm_best.pt"
  DATA="outputs/data/subsets/trajs_${PCT}.h5"
  OUT="outputs/sfe_policy_${PCT}"
  echo "----- IMAGINE ${PCT}% ckpt=${CKPT} -> ${OUT} $(date) -----" | tee -a "$MASTER"
  if [ ! -f "$CKPT" ]; then
    echo "----- 跳过 ${PCT}%：世界模型权重不存在 -----" | tee -a "$MASTER"
    continue
  fi
  OMP_NUM_THREADS=1 $PY scripts/train_imagine_policy.py \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --epochs 400 --batch-size 128 --horizon 12 \
    --entropy-coef 2e-2 --lr-actor 1.5e-4 \
    --out-dir "$OUT" \
    > "outputs/sfe_imagine_${PCT}.log" 2>&1
  echo "----- IMAGINE ${PCT}% 退出码=$? $(date) -----" | tee -a "$MASTER"
done
echo "========== 全部想象策略子集训练完成 $(date) ==========" | tee -a "$MASTER"
