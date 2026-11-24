#!/bin/bash
# 消融实验 统一评估：4 组消融策略跑标准环境(30回合) + 最优策略跑泛化场景(50场景)。mujoco 环境。
set -u
cd /data/wzh/robot/vla-world-model-lab
PY=/opt/miniconda3/envs/mujoco/bin/python
MASTER=outputs/ablation_eval_master.log
echo "========== 消融实验 消融评估开始 $(date) ==========" > "$MASTER"

eval_std () {
  local TAG=$1
  local POL="outputs/abl_${TAG}/imagine_policy.pt"
  echo "----- EVAL STD ${TAG} $(date) -----" | tee -a "$MASTER"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY scripts/eval_ablation.py \
    --policy "$POL" --mode standard --episodes 30 --device cpu \
    --out "outputs/abl_${TAG}/eval_standard.json" 2>&1 | tee -a "$MASTER"
}

eval_std h5
eval_std h10
eval_std h20
eval_std norew

# 泛化测试：满数据最优策略 v2（与项目已有 benchmark 同场景口径）
echo "----- EVAL GENERALIZE v2 $(date) -----" | tee -a "$MASTER"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY scripts/eval_ablation.py \
  --policy outputs/imagine_policy_v2/imagine_policy.pt --mode generalize --device cpu \
  --out outputs/gen_imagine_v2/eval_generalize.json 2>&1 | tee -a "$MASTER"

echo "========== 消融实验 消融评估完成 $(date) ==========" | tee -a "$MASTER"
