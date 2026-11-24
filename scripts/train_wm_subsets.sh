#!/bin/bash
# 世界模型样本效率实验：按 10/30/50/100% 离线数据各训一个世界模型。
# 严格沿用最优检查点配置（stoch48/det200/batch16/kl1.5/freebits0.05/curriculum30），
# 唯一变量为训练数据量，保证受控对比。
set -u
cd /data/wzh/robot/vla-world-model-lab
PY=/opt/miniconda3/envs/vla/bin/python
MASTER=outputs/sfe_wm_master.log
echo "========== 世界模型子集训练开始 $(date) ==========" > "$MASTER"

for PCT in 10 30 50 100; do
  DATA="outputs/data/subsets/trajs_${PCT}.h5"
  OUT="outputs/sfe_wm_${PCT}"
  echo "----- WM ${PCT}% data=${DATA} -> ${OUT} $(date) -----" | tee -a "$MASTER"
  OMP_NUM_THREADS=1 $PY trainer/train_world_model.py \
    --data "$DATA" \
    --output-dir "$OUT" \
    --stoch-dim 48 --det-dim 200 --latent-dim 64 \
    --group-lr --batch-size 16 \
    --seq-len-start 10 --curriculum-steps 30 --seq-len 20 \
    --eval-interval 500 --eval-trajs 4 \
    --max-steps 5000 \
    --kl-scale 1.5 --free-bits 0.05 \
    --reward-scale 2.0 --value-scale 1.0 \
    --device cuda --seed 0 \
    > "outputs/sfe_wm_${PCT}.log" 2>&1
  echo "----- WM ${PCT}% 退出码=$? $(date) -----" | tee -a "$MASTER"
done
echo "========== 全部世界模型子集训练完成 $(date) ==========" | tee -a "$MASTER"
