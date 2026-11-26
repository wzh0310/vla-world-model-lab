"""用 LeRobot 官方接口 + Parquet 列校验转换后数据集的字段/时序/划分/图像/指令映射。

时序校验直接读 Parquet 列（快）；图像解码、字段完整性等用 LeRobotDataset[i] 抽样验证，
避免逐帧解码 900+ 帧导致的超时。
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REQUIRED_FIELDS = {
    "episode_index", "frame_index", "timestamp",
    "observation.images.main", "observation.language_instruction", "action",
}
DEFAULT_FIELDS = {"index", "task_index", "task"}
FPS = 50


def read_parquet(root):
    """读取 split 目录下所有 data parquet 并拼接。"""
    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/wzh/robot/datasets/vla_pushcube_25/lerobot")
    args = ap.parse_args()
    root = args.root

    checks = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        checks.append((name, status, detail))
        print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))

    summary = json.load(open(os.path.join(root, "conversion_summary.json"), encoding="utf-8"))
    check("conversion_summary 存在", summary.get("format") == "lerobot_v0.7.0")

    expected = {"train": (20, 745), "val": (5, 183)}
    loaded = {}
    for split in ("train", "val"):
        repo = f"wzh/pushcube_vla_{split}"
        rroot = os.path.join(root, split)
        check(f"{split} 目录存在", os.path.isdir(rroot))

        # --- Parquet 列级校验（快） ---
        df = read_parquet(rroot)
        n_ep, n_fr = expected[split]
        check(f"{split} Parquet 帧数", len(df) == n_fr, f"rows={len(df)} expect={n_fr}")
        parquet_fields = set(df.columns) - {"task"}  # task 由 reader 注入
        check(f"{split} Parquet 必需列", (REQUIRED_FIELDS - {"task"}).issubset(parquet_fields),
              f"missing={sorted((REQUIRED_FIELDS - {'task'}) - parquet_fields)}")

        eps = df["episode_index"].astype(int).values
        fis = df["frame_index"].astype(int).values
        tss = df["timestamp"].astype(float).values
        uniq_eps = sorted(set(eps.tolist()))
        check(f"{split} episode 数", len(uniq_eps) == n_ep, f"episodes={len(uniq_eps)}")

        # 每回合内 frame_index 从 0 连续递增；timestamp 从 0 起、步长 1/fps
        per_ep_ok, ts_ok, ts_step_ok = True, True, True
        for ep in uniq_eps:
            mask = eps == ep
            fi = fis[mask]
            ts = tss[mask]
            if not np.array_equal(fi, np.arange(len(fi))):
                per_ep_ok = False
            if abs(float(ts[0])) > 1e-4:
                ts_ok = False
            steps = np.diff(ts)
            if len(steps) and not np.allclose(steps, 1.0 / FPS, atol=1e-3):
                ts_step_ok = False
        check(f"{split} 每回合 frame_index 从0连续递增", per_ep_ok)
        check(f"{split} 每回合 timestamp 从0起", ts_ok)
        check(f"{split} timestamp 步长=1/{FPS}", ts_step_ok,
              f"first ep ts[:4]={[round(t,3) for t in tss[:4]]}")

        # index 全局连续；episode_index 非递减（回合按序块排列）
        idx = df["index"].astype(int).values
        check(f"{split} index 全局连续", np.array_equal(idx, np.arange(len(idx))))
        check(f"{split} episode 按块排列",
              all(eps[i] <= eps[i+1] for i in range(len(eps)-1)))

        # 回合级指令帧内一致 + task_index 映射
        instr_col = "observation.language_instruction"
        instr_by_ep = df.groupby("episode_index")[instr_col].nunique()
        check(f"{split} 回合级指令帧内一致", bool((instr_by_ep == 1).all()),
              f"max_unique_per_ep={int(instr_by_ep.max())}")
        check(f"{split} 指令非空", bool(df[instr_col].astype(str).str.strip().ne("").all()))
        tis = sorted(set(df["task_index"].astype(int).tolist()))
        check(f"{split} task_index 映射存在", len(tis) > 0 and min(tis) >= 0,
              f"count={len(tis)} indices={tis[:8]}")

        # 动作列范围（全量，快）
        acts = np.vstack(df["action"].values)
        check(f"{split} 动作维度=4", acts.shape[1] == 4, f"{acts.shape}")
        check(f"{split} 动作范围[-1,1]",
              -1.0001 <= float(acts.min()) and float(acts.max()) <= 1.0001,
              f"[{acts.min():.4f}, {acts.max():.4f}]")

        # --- 官方接口抽样校验（图像解码 / 字段 / prompt 映射） ---
        ds = LeRobotDataset(repo_id=repo, root=rroot)
        loaded[split] = ds
        check(f"{split} 官方加载帧数", len(ds) == n_fr, f"len={len(ds)}")
        check(f"{split} 官方 num_episodes", ds.num_episodes == n_ep)

        row = ds[0]
        keys = set(row.keys())
        check(f"{split} 官方字段齐全",
              REQUIRED_FIELDS.issubset(keys) and DEFAULT_FIELDS.issubset(keys),
              f"missing={sorted((REQUIRED_FIELDS | DEFAULT_FIELDS) - keys)}")
        img = row["observation.images.main"]
        check(f"{split} 图像解码形状", tuple(img.shape) == (3, 224, 224), f"{tuple(img.shape)}")
        check(f"{split} 指令与 task 一致", str(row["task"]) == str(row[instr_col]))
        # 抽 3 个不同位置的帧确认图像可稳定解码
        sample_ok = True
        for i in {0, len(ds)//2, len(ds)-1}:
            try:
                _ = ds[i]["observation.images.main"]
            except Exception:
                sample_ok = False
        check(f"{split} 抽样帧图像可解码", sample_ok)

    # 划分无重叠
    tr_src = set(summary["splits"][0]["source_episode_ids"])
    va_src = set(summary["splits"][1]["source_episode_ids"])
    check("训练/验证源 episode 无重叠", tr_src.isdisjoint(va_src) and len(tr_src | va_src) == 25,
          f"train={len(tr_src)}, val={len(va_src)}")

    total = len(checks)
    passed = sum(1 for _, s, _ in checks if s == "PASS")
    print(f"\n结果: {passed}/{total} 通过")
    for name, status, detail in checks:
        if status == "FAIL":
            print(f"  FAILED -> {name} {detail}")
    out = {
        "status": "PASS" if passed == total else "FAIL",
        "checks": [{"name": n, "status": s, "detail": d} for n, s, d in checks],
        "train": {"episodes": loaded["train"].num_episodes, "frames": len(loaded["train"])},
        "val": {"episodes": loaded["val"].num_episodes, "frames": len(loaded["val"])},
        "fields": sorted(REQUIRED_FIELDS | DEFAULT_FIELDS),
        "note": "frame_index/timestamp 按官方规范每回合从 0 重排",
    }
    with open(os.path.join(root, "verify_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
