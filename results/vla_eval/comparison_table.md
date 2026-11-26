# 微调模型 vs 官方零样本：基础效果对比数据表

同一评估脚本（`vla/eval/eval.py`）、同一 PushCube 环境、同一成功判据（方块中心距红盘 < 0.03 m）、同为 int4 量化、最大 40 步。测试集 5 条代表性标准指令 × 每指令 3 回合 = 15 回合。

> 说明：环境 `randomize_start=False`，同一指令不同 seed 初始状态相同，轨迹一致；每指令 3 回合为固定场景重复验证，非独立泛化样本。

## 总体

| 模式 | 权重 | unnorm_key | 成功率 | 平均距离改善 | 加载显存 |
| --- | --- | --- | --- | --- | --- |
| 微调后 | OpenVLA-7B int4 + LoRA adapter（step192, eval_loss=1.9890） | pushcube_vla | 3/15 = 0.20 | 0.0545 m | 4.58 GB |
| 官方零样本 | OpenVLA-7B int4（底座） | bridge_orig | 0/15 = 0.00 | 0.0 m | 4.55 GB |

## 分指令明细（微调后）

| 指令 | 成功回合 | 距离 起始→结束（最小） |
| --- | --- | --- |
| push the cube into the red marked area | 3/3 | 0.2000 → 0.0216 (0.0216) |
| push the cube to the red target | 0/3 | 0.2000 → 0.3251 (0.0954) |
| move block to red target | 0/3 | 0.2000 → 0.1368 (0.1156) |
| slide the cube ahead to the red goal | 0/3 | 0.2000 → 0.1395 (0.1240) |
| push the block onto the red disc | 0/3 | 0.2000 → 0.1044 (0.0889) |

## 分指令明细（官方零样本）

5 条指令均 0/3；15/15 回合距离保持 0.2000 不变（模型未建立有效接触）。

## 结论边界

- 微调带来可测量的指令跟随：从「完全不动」到「能推动方块、方向正确，并在一条指令上成功停靠 3 次」。
- 总体成功率 0.20，存在过冲且不会修正，早期 checkpoint 控制精度有限；不构成任务学会或收敛结论。
- 演示视频（仓库外）：成功案例 `demo_video_success/demo.mp4`（距离 0.2→0.0265）、标准指令过冲案例 `demo_video/demo.mp4`（0.2→0.3251）。

原始数据：`finetuned.json`、`zeroshot.json`。
