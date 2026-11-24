#!/bin/bash
# 消融实验评估编排：统一口径评估 4 组模型的多步预测
set -e
cd /data/wzh/robot/vla-world-model-lab
P=/data/lxt/miniconda3/envs/py312-robot/bin/python

TRAJS="traj_0002,traj_0007,traj_0011,traj_0015,traj_0020"
STARTS="2,8"
COMMON="--trajs $TRAJS --starts $STARTS --steps 20 --key-steps 1,5,10,20"

echo "=== 评估 基线(有KL,stoch64) ==="
$P trainer/eval_multistep_full.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt --out-dir outputs/abl_base $COMMON 2>&1 | tee outputs/abl_base_eval.log

echo "=== 评估 stoch16 ==="
$P trainer/eval_multistep_full.py --ckpt outputs/abl_stoch16/checkpoints/wm_step004000.pt --out-dir outputs/abl_stoch16 $COMMON --vis-trajs "" 2>&1 | tee outputs/abl_stoch16_eval.log

echo "=== 评估 stoch32 ==="
$P trainer/eval_multistep_full.py --ckpt outputs/abl_stoch32/checkpoints/wm_step004000.pt --out-dir outputs/abl_stoch32 $COMMON --vis-trajs "" 2>&1 | tee outputs/abl_stoch32_eval.log

echo "=== 评估 无KL ==="
$P trainer/eval_multistep_full.py --ckpt outputs/abl_nokl/checkpoints/wm_step004000.pt --out-dir outputs/abl_nokl $COMMON --vis-trajs "" 2>&1 | tee outputs/abl_nokl_eval.log

echo "=== 全部评估完成 ==="
