"""数据集质量抽查：随机抽 3 条轨迹逐帧核对图像/动作/指令匹配。

检查维度：
  1. 图像-动作-指令对齐：帧数一致、指令回合级一致、动作帧索引连续
  2. 动作异常值：min-max 归一化后越界、突变（相邻帧差超阈值）、NaN/Inf
  3. 图像模糊：Laplacian 方差（清晰度）阈值检测
  4. 指令错位：指令是否属于模板库且类别与 template_id 一致

同时给出全数据集的动作范围与图像清晰度分布作为统计基线。
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/data/wzh/robot/vla-world-model-lab")
from vla.data.instruction_templates import TEMPLATES, get_category, get_instruction  # noqa: E402

TEMPLATE_TO_INSTR = {tid: ins for tid, (cat, ins) in TEMPLATES.items()}
INSTR_TO_TID = {ins: tid for tid, ins in TEMPLATE_TO_INSTR.items()}


def laplacian_var(img_chw_float):
    """图像清晰度：Laplacian 方差越大越清晰。输入 CHW float[0,1]。"""
    arr = np.asarray(img_chw_float)
    if arr.ndim == 3 and arr.shape[0] == 3:
        hwc = (np.transpose(np.clip(arr, 0, 1), (1, 2, 0)) * 255).astype(np.uint8)
    else:
        hwc = arr.astype(np.uint8)
    gray = cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/wzh/robot/datasets/vla_pushcube_25/preprocessed")
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--n-audit", type=int, default=3)
    ap.add_argument("--out", default="/data/wzh/robot/datasets/vla_pushcube_25/quality_audit_v1.json")
    args = ap.parse_args()

    tr = np.load(os.path.join(args.root, "train.npz"), allow_pickle=True)
    va = np.load(os.path.join(args.root, "val.npz"), allow_pickle=True)
    images = np.concatenate([tr["images"], va["images"]], axis=0)
    actions = np.concatenate([tr["actions"], va["actions"]], axis=0)
    instrs = np.concatenate([tr["instructions"], va["instructions"]], axis=0)
    ep_ids = np.concatenate([tr["episode_ids"], va["episode_ids"]], axis=0)
    frame_idx = np.concatenate([tr["frame_indices"], va["frame_indices"]], axis=0)
    template_ids = np.concatenate([tr["template_ids"], va["template_ids"]], axis=0)
    cats = np.concatenate([tr["categories"], va["categories"]], axis=0)

    # 反归一化到物理空间做异常判据（归一化空间跳变 2.0 只是 min-max 映射假象）
    ns = json.load(open(os.path.join(args.root, "norm_stats.json"), encoding="utf-8"))
    lo = np.asarray(ns["action"]["min"], dtype=np.float64)
    hi = np.asarray(ns["action"]["max"], dtype=np.float64)
    span = hi - lo
    actions_phys = lo + (actions.astype(np.float64) + 1.0) * span / 2.0
    MAX_STEP_M = 0.02  # 单步量程上限（pushcube_action.yaml）

    results = {"seed": args.seed, "audited_episodes": [], "global_stats": {}, "issues": []}

    # ---- 全局统计基线 ----
    results["global_stats"] = {
        "total_frames": int(len(images)),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "action_mean_abs": np.abs(actions).mean(axis=0).tolist(),
        "action_out_of_range_frames": int(np.sum((actions < -1.0001) | (actions > 1.0001))),
        "action_nonfinite_frames": int(np.sum(~np.isfinite(actions))),
        "action_physical_max_step_m": float(np.abs(np.diff(actions_phys, axis=0)).max()),
        "action_physical_abnormal_steps": int(np.sum(
            np.abs(np.diff(actions_phys, axis=0))[:, :3] > MAX_STEP_M + 1e-6)),
        "normalization_note": "归一化跳变2.0为min-max映射假象；物理空间跳变<=0.02m量程即正常",
        "image_sharpness_mean": float(np.mean([laplacian_var(im) for im in images[::20]])),
        "image_sharpness_min": float(np.min([laplacian_var(im) for im in images[::20]])),
        "instruction_unique": int(len(set(str(x) for x in instrs))),
    }

    # 指令错位：每条指令是否属于模板库且类别一致
    mismatches = []
    for i in range(len(instrs)):
        tid = str(template_ids[i])
        ins = str(instrs[i])
        cat = str(cats[i])
        if tid not in TEMPLATE_TO_INSTR or TEMPLATE_TO_INSTR[tid] != ins:
            mismatches.append({"frame": i, "tid": tid, "instr": ins, "reason": "tid-instr mismatch"})
        elif get_category(tid) != cat:
            mismatches.append({"frame": i, "tid": tid, "cat": cat, "reason": "category mismatch"})
    results["global_stats"]["instruction_template_mismatches"] = len(mismatches)
    if mismatches:
        results["issues"].append({"type": "instruction_mismatch", "count": len(mismatches),
                                  "samples": mismatches[:5]})

    # ---- 随机抽查 N 条轨迹 ----
    rng = np.random.default_rng(args.seed)
    all_eps = sorted(set(int(e) for e in ep_ids))
    picked = sorted(rng.choice(all_eps, size=args.n_audit, replace=False).tolist())
    for ep in picked:
        idxs = np.where(ep_ids == ep)[0]
        idxs = sorted(idxs, key=lambda i: int(frame_idx[i]))
        ep_instrs = set(str(instrs[i]) for i in idxs)
        ep_tids = set(str(template_ids[i]) for i in idxs)
        ep_actions = actions[idxs]
        ep_phys = actions_phys[idxs]
        # 动作突变：物理空间相邻帧差（单步量程 0.02m 为正常上限）
        jumps = np.abs(np.diff(ep_phys, axis=0))[:, :3]
        max_jump = float(jumps.max()) if len(jumps) else 0.0
        jump_frames = int(np.sum(jumps > MAX_STEP_M + 1e-6))
        # 清晰度
        sharp = [laplacian_var(images[i]) for i in idxs]
        blurry = int(sum(1 for s in sharp if s < 5.0))
        audit = {
            "episode_id": ep,
            "frames": len(idxs),
            "instruction": list(ep_instrs)[0] if len(ep_instrs) == 1 else list(ep_instrs),
            "template_id": list(ep_tids)[0] if len(ep_tids) == 1 else list(ep_tids),
            "instruction_consistent": len(ep_instrs) == 1 and len(ep_tids) == 1,
            "frame_index_continuous": bool(np.array_equal(frame_idx[idxs], np.arange(len(idxs)))),
            "action_range": [ep_actions.min(axis=0).tolist(), ep_actions.max(axis=0).tolist()],
            "action_max_step_jump_m": round(max_jump, 5),
            "action_abnormal_frames": jump_frames,
            "image_sharpness_min": round(float(min(sharp)), 2),
            "image_sharpness_mean": round(float(np.mean(sharp)), 2),
            "blurry_frames": blurry,
            "verdict": "PASS",
        }
        if (not audit["instruction_consistent"] or not audit["frame_index_continuous"]
                or jump_frames > 0 or blurry > len(idxs) * 0.2):
            audit["verdict"] = "REVIEW"
            results["issues"].append({"type": "episode_review", "episode_id": ep})
        results["audited_episodes"].append(audit)

    results["status"] = "PASS" if not any(
        i["type"] == "instruction_mismatch" for i in results["issues"]
    ) else "FAIL"
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[质量抽查] seed={args.seed} 轨迹={picked}")
    for a in results["audited_episodes"]:
        print(f"  ep{a['episode_id']}: {a['frames']}帧 指令一致={a['instruction_consistent']} "
              f"帧连续={a['frame_index_continuous']} 物理最大跳变={a['action_max_step_jump_m']}m "
              f"异常步={a['action_abnormal_frames']} 模糊帧={a['blurry_frames']} -> {a['verdict']}")
    g = results["global_stats"]
    print(f"[全局] 指令模板错位={g['instruction_template_mismatches']} "
          f"动作越界={g['action_out_of_range_frames']} 非有限={g['action_nonfinite_frames']} "
          f"清晰度均值={g['image_sharpness_mean']:.2f}")
    print(f"[结果] {results['status']} -> {args.out}")


if __name__ == "__main__":
    main()
