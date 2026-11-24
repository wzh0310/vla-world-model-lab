#!/bin/bash
# 监控消融训练完成，自动触发评估与汇总
cd /data/wzh/robot/vla-world-model-lab
LOG=outputs/ablation_train_master.log
# 等待训练 master 出现"全部训练完成"标记（最多等 40 分钟）
for i in $(seq 1 240); do
  if grep -q "全部训练完成" "$LOG" 2>/dev/null; then
    echo "[WATCH] 训练完成，启动评估 $(date)" >> outputs/ablation_watch.log
    bash scripts/ablation_eval.sh >> outputs/ablation_watch.log 2>&1
    echo "[WATCH] 评估完成，启动汇总 $(date)" >> outputs/ablation_watch.log
    /data/lxt/miniconda3/envs/py312-robot/bin/python trainer/ablation_report.py >> outputs/ablation_watch.log 2>&1
    echo "[WATCH] 全部完成 $(date)" >> outputs/ablation_watch.log
    exit 0
  fi
  sleep 10
done
echo "[WATCH] 超时退出 $(date)" >> outputs/ablation_watch.log
