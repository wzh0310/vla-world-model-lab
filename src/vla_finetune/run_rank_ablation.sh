#!/bin/bash
# 固定预算 LoRA 秩消融：只变 rank/alpha（保持 alpha/r=2），其余完全一致。
# 串行训练三组，各用同一个 eval.py 评估闭环成功率。
set +e
cd /data/wzh/robot/vla-world-model-lab
export MUJOCO_GL=egl
BASE=results/vla_ablation
mkdir -p "$BASE"
PY=/opt/miniconda3/envs/vla/bin/python
COMMON="--config vla/configs/trainer_lora.yaml --learning-rate 0.0001 --max-grad-norm 1.0 --max-steps 192"

echo "[ablation] start $(date '+%F %T')" | tee -a "$BASE/ablation_driver.log"
for R in 4 8 16; do
  A=$((2*R))
  D="$BASE/rank${R}"
  mkdir -p "$D"
  echo "[ablation] TRAIN rank=$R alpha=$A steps=192 $(date '+%T')" | tee -a "$BASE/ablation_driver.log"
  $PY src/vla_finetune/train.py $COMMON --lora-rank $R --lora-alpha $A --out-dir "$D" \
      > "$D/train.log" 2>&1
  echo "[ablation] TRAIN rank=$R exit=$? $(date '+%T')" | tee -a "$BASE/ablation_driver.log"
  if [ -d "$D/best_adapter" ]; then
    echo "[ablation] EVAL rank=$R $(date '+%T')" | tee -a "$BASE/ablation_driver.log"
    MUJOCO_GL=egl $PY vla/eval/eval.py --mode finetuned --adapter "$D/best_adapter" \
        --episodes 3 --max-steps 40 --out "$D/eval.json" > "$D/eval.log" 2>&1
    echo "[ablation] EVAL rank=$R exit=$? $(date '+%T')" | tee -a "$BASE/ablation_driver.log"
  else
    echo "[ablation] rank=$R 无 best_adapter，跳过评估" | tee -a "$BASE/ablation_driver.log"
  fi
done
echo "[ablation] ALL DONE $(date '+%F %T')" | tee -a "$BASE/ablation_driver.log"
echo DONE > "$BASE/DONE"
