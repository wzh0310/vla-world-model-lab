"""校验 v1.0 数据集：增强生效、验证集不变、字段完整、动作范围、指令同义映射。

全部用 numpy 向量化，避免逐帧/逐块 Python 循环导致大 npz 加载后超时。
"""

import argparse
import json
import os

import numpy as np

TEMPLATE_TO_INSTR = {
    "D01": "push the cube to the red target area", "D02": "push the cube onto the red target",
    "D03": "move the cube to the red target disc", "D04": "push the blue cube to the red circle",
    "B01": "push block to goal", "B02": "push cube to target", "B03": "move block to red disc",
    "B04": "push cube onto red goal",
    "S01": "move the blue cube on the desk to the circular mark",
    "S02": "slide the blue cube across the table to the red circular target",
    "S03": "push the small cube on the tabletop into the red marked area",
    "S04": "move the blue block on the desk onto the red round marker",
    "V01": "slide the cube forward to the red region", "V02": "push the block ahead into the red target zone",
    "V03": "shift the cube toward the red goal area", "V04": "drive the cube to the red circular spot",
    "R01": "push the cube away from its starting position toward the red target",
    "R02": "move the cube off its initial spot and into the red target area",
    "R03": "push the block from where it starts to the red circular mark",
    "R04": "slide the cube out of its starting position onto the red goal",
}
PARAPHRASES = {
    "D01": ["push the cube into the red marked area", "move the block to the red target region"],
    "D02": ["place the cube on the red target", "push the block onto the red circle"],
    "D03": ["move the cube onto the red target disc", "push the blue cube to the red circular mark"],
    "D04": ["push the blue block to the red circle", "move the cube onto the red target spot"],
    "B01": ["push block to red goal", "move block onto the target"],
    "B02": ["push cube to goal", "move cube to red target"],
    "B03": ["move block to red target", "push block onto red disc"],
    "B04": ["push cube to red goal", "move cube onto red target"],
    "S01": ["slide the blue cube on the desk to the circular mark",
            "move the small blue block across the table to the red circle"],
    "S02": ["slide the blue block across the desk to the red circular target",
            "move the blue cube over the tabletop to the red round mark"],
    "S03": ["push the small cube on the table into the red marked circle",
            "slide the small block on the tabletop onto the red target area"],
    "S04": ["move the blue block on the desk to the red circular marker",
            "push the blue cube across the desk onto the red round target"],
    "V01": ["slide the cube forward into the red region", "shift the block ahead to the red target"],
    "V02": ["push the block forward into the red target zone", "slide the cube ahead to the red goal"],
    "V03": ["move the cube toward the red goal area", "shift the block into the red target region"],
    "V04": ["drive the block to the red circular target", "push the cube onto the red circular spot"],
    "R01": ["move the cube away from its start toward the red target",
            "push the block from its initial position to the red goal"],
    "R02": ["slide the cube off its starting spot into the red target area",
            "move the block away from its initial position to the red goal"],
    "R03": ["push the cube from where it begins to the red circular mark",
            "move the block from its starting spot to the red round target"],
    "R04": ["slide the block out of its initial spot onto the red goal",
            "push the cube away from the start into the red target area"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10", default="/data/wzh/robot/datasets/vla_pushcube_25/v1.0")
    ap.add_argument("--src", default="/data/wzh/robot/datasets/vla_pushcube_25/preprocessed")
    args = ap.parse_args()

    checks = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        checks.append((name, status, detail))
        print(f"[{status}] {name}" + (f" | {detail}" if detail else ""), flush=True)

    tr = np.load(os.path.join(args.v10, "train_v1.npz"), allow_pickle=True)
    va = np.load(os.path.join(args.v10, "val_v1.npz"), allow_pickle=True)
    va_src = np.load(os.path.join(args.src, "val.npz"), allow_pickle=True)

    need = ["images", "actions", "instructions", "episode_ids", "frame_indices",
            "task_indices", "categories", "template_ids", "aug_types", "source_episode_ids"]
    check("训练集字段完整", all(k in tr.files for k in need))
    aug_types = tr["aug_types"].astype(str)
    check("增强类型齐全", set(aug_types) == {"original", "image_aug", "instr_aug_1", "instr_aug_2"},
          f"{sorted(set(aug_types))}")

    check("训练集帧数=2980", len(tr["images"]) == 2980, f"{len(tr['images'])}")
    check("源轨迹数=20", len(set(int(x) for x in tr["source_episode_ids"])) == 20)
    tids = tr["template_ids"].astype(str)
    check("template 覆盖=16", len(set(tids)) == 16, f"unique={len(set(tids))}")

    # 用 np.unique + 一次 groupby 统计每 (ep,aug_type) 块，替代 80 次 np.where
    src = tr["source_episode_ids"].astype(np.int64)
    keys = src * 10 + np.searchsorted(np.array(["image_aug", "instr_aug_1", "instr_aug_2", "original"]), aug_types)
    uniq_keys, counts = np.unique(keys, return_counts=True)
    check("每轨迹4种增强块", len(uniq_keys) == 80,
          f"blocks={len(uniq_keys)} (20 episodes x 4 aug types)")

    # 指令同义映射：一次性向量化比对（每帧按 tid+aug_type 查期望指令）
    instrs = tr["instructions"].astype(str)
    expected = np.empty(len(instrs), dtype=object)
    for tid in set(tids):
        para = PARAPHRASES.get(tid, ["", ""])
        for at in ["original", "image_aug", "instr_aug_1", "instr_aug_2"]:
            mask = (tids == tid) & (aug_types == at)
            val = TEMPLATE_TO_INSTR[tid] if at in ("original", "image_aug") else (
                para[0] if at == "instr_aug_1" else para[1])
            expected[mask] = val
    mapping_ok = bool(np.all(instrs == expected))
    check("指令同义映射正确", mapping_ok,
          f"mismatch={int(np.sum(instrs != expected))}")

    # 图像增强差异（用 ep0 的 original vs image_aug 块，一次布尔索引）
    mask_o = (src == 0) & (aug_types == "original")
    mask_i = (src == 0) & (aug_types == "image_aug")
    if mask_o.sum() and mask_i.sum():
        diff = float(np.mean(np.abs(tr["images"][mask_o].astype(np.float32) -
                                    tr["images"][mask_i].astype(np.float32))))
        check("图像增强生效", diff > 1e-4, f"mean_abs_diff={diff:.4f}")
        check("图像增强幅度轻微", diff < 0.15, f"mean_abs_diff={diff:.4f}")
        check("指令增强图像不变", bool(np.array_equal(
            tr["images"][mask_o], tr["images"][(src == 0) & (aug_types == "instr_aug_1")])))
        check("动作增强前后不变", bool(np.array_equal(
            tr["actions"][mask_o], tr["actions"][mask_i])))

    acts, imgs = tr["actions"], tr["images"]
    check("动作范围[-1,1]", -1.0001 <= float(acts.min()) and float(acts.max()) <= 1.0001,
          f"[{acts.min():.4f},{acts.max():.4f}]")
    check("图像范围[0,1]", 0.0 <= float(imgs.min()) and float(imgs.max()) <= 1.0,
          f"[{imgs.min():.4f},{imgs.max():.4f}]")

    check("验证集帧数不变", len(va["images"]) == len(va_src["images"]) == 183)
    check("验证集图像逐字节相同", bool(np.array_equal(va["images"], va_src["images"])))
    check("验证集动作逐字节相同", bool(np.array_equal(va["actions"], va_src["actions"])))
    check("验证集指令逐字节相同", bool(np.array_equal(va["instructions"], va_src["instructions"])))

    base = os.path.dirname(args.v10)
    check("quality_audit_v1.json 存在", os.path.exists(os.path.join(base, "quality_audit_v1.json")))
    check("stats_report.json 存在", os.path.exists(os.path.join(args.v10, "stats_report.json")))

    total = len(checks)
    passed = sum(1 for _, s, _ in checks if s == "PASS")
    print(f"\n结果: {passed}/{total} 通过", flush=True)
    out = {"status": "PASS" if passed == total else "FAIL",
           "checks": [{"name": n, "status": s, "detail": d} for n, s, d in checks],
           "train_frames": int(len(tr["images"])), "val_frames": int(len(va["images"])),
           "note": "训练集 20 源轨迹按分层划分覆盖 16 个模板，另 4 个在验证集"}
    with open(os.path.join(args.v10, "verify_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
