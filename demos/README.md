# demos

可入库的演示素材（`outputs/` 不入 git，本目录为精选副本）。

## prediction/（开环预测演示）

| 文件 | 内容 |
|---|---|
| `real_vs_pred20_traj0002.gif` | 真实 vs 20 步开环预测并排动图（基线 stoch64+KL，仅首帧+动作序列，全程不接真实帧） |
| `real_vs_pred20_traj0002_grid.png` | 同上的静态 2×20 网格（上排真实 / 下排预测） |
| `infer_traj0002_pred.gif` | 一键推理脚本 `trainer/infer_video.py` 产物：traj_0002 起点 8，20 步预测 |
| `infer_traj0002_compare.png` | 一键推理静态对比网格 |
| `psnr_decay_base.png` | 基线模型 20 步开环 PSNR/SSIM 衰减曲线 |
| `rollout_test_traj0002.gif` / `.png` | `WorldModel.rollout()` 管线验证：前 3 帧热身 + 10 步开环预测（PSNR 32.0→23.2 dB，无崩坏） |

## ablation/（消融实验图表）

| 文件 | 内容 |
|---|---|
| `psnr_by_dim.png` | 随机态维度消融（stoch 16/32/64）PSNR 随步长曲线 |
| `psnr_by_kl.png` | KL 消融（有/无）PSNR 随步长曲线 |
| `ssim_compare.png` | 两组消融 SSIM 对比 |
| `stability_bar.png` | 长程稳定性柱状图（1→20 PSNR 衰减 / step20 SSIM） |

数据与结论详见 `docs/消融实验与工程优化报告.md`；逐 step 原始数值在
`outputs/ablation_report/metrics/`（不入 git，可由 `scripts/ablation_eval.sh` 复现）。

## 复现方式

```bash
# 一键推理演示
python trainer/infer_video.py --ckpt outputs/wm_tune1/checkpoints/wm_step004000.pt \
    --traj traj_0002 --start 8 --steps 20 --out outputs/infer_demo
# 消融图表
bash scripts/ablation_train.sh && bash scripts/ablation_eval.sh && python trainer/ablation_report.py
```
