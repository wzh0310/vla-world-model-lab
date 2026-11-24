#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总样本效率实验的 8 个评估结果 -> results/sfe_results.json。

PPO 四档来自 sfe_eval_master.log 的 [SFE] 行；WM 四档来自 sfe_eval_wm_{pct}.json。
这里统一从 eval_unified_sfe.py --all 的输出读取（若存在），否则从分散文件归集。
"""
import json
import os
import re

ROOT = "/data/wzh/robot/vla-world-model-lab"
RES = os.path.join(ROOT, "results")
MASTER = os.path.join(ROOT, "outputs/sfe_eval_master.log")


def main():
    rows = []
    # 优先：从 master log 解析所有 [SFE] 成功行（每个含 JSON 前的关键字段）
    # 但 log 行是非结构化打印，改为让 eval 脚本逐档写 json。
    # PPO: 逐档写 outputs/sfe_eval_ppo_{step}.json（由本汇总前确保）
    for step in [200000, 600000, 1000000, 2000000]:
        p = os.path.join(ROOT, f"outputs/sfe_eval_ppo_{step}.json")
        if os.path.exists(p):
            m = json.load(open(p))
            m.update({"group": "ppo", "label": f"PPO-vision-{step//1000}k", "data_level": step})
            rows.append(m)
    for pct in [10, 30, 50, 100]:
        p = os.path.join(ROOT, f"outputs/sfe_eval_wm_{pct}.json")
        if os.path.exists(p):
            m = json.load(open(p))
            m.update({"group": "wm", "label": f"WorldModel-imagine-{pct}%", "data_level": pct})
            rows.append(m)

    out = os.path.join(RES, "sfe_results.json")
    os.makedirs(RES, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[COLLECT] {len(rows)} 条结果 -> {out}")
    for r in rows:
        print(f"  {r['label']:26s} succ={r['success_rate']:.0%} "
              f"reward={r['mean_reward']:+.4f} dist={r['mean_final_dist']:.4f}")


if __name__ == "__main__":
    main()
