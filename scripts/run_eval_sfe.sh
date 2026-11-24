#!/bin/bash
# 样本效率统一评估：PPO 四档(cuda) + 世界模型四档(cpu 单线程)，各 30 回合。
set -u
cd /data/wzh/robot/vla-world-model-lab
PY=/opt/miniconda3/envs/mujoco/bin/python
MASTER=outputs/sfe_eval_master.log
echo "========== 样本效率统一评估开始 $(date) ==========" > "$MASTER"
mkdir -p results

# PPO 对照组（cuda，快）
for STEP in 200000 600000 1000000 2000000; do
  echo "----- EVAL PPO ${STEP} $(date) -----" | tee -a "$MASTER"
  OMP_NUM_THREADS=1 $PY scripts/eval_unified_sfe.py --group ppo --step $STEP \
    --episodes 30 --device cuda --out "outputs/sfe_eval_ppo_${STEP}.json" 2>&1 | tee -a "$MASTER"
done

# 世界模型实验组（cpu 单线程，避免多线程拖慢）
for PCT in 10 30 50 100; do
  echo "----- EVAL WM ${PCT}% $(date) -----" | tee -a "$MASTER"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY scripts/eval_unified_sfe.py --group wm --pct $PCT \
    --episodes 30 --device cpu \
    --out "outputs/sfe_eval_wm_${PCT}.json" 2>&1 | tee -a "$MASTER"
done

echo "========== 样本效率统一评估完成 $(date) ==========" | tee -a "$MASTER"
