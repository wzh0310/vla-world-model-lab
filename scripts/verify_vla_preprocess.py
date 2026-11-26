"""校验预处理数据集：字段/形状/范围、分层划分、归一化可逆、Dataset+DataLoader 加载。"""

import argparse
import importlib.util
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = "/data/wzh/robot/vla-world-model-lab"
DATASET_PATH = os.path.join(REPO, "vla", "data", "pushcube_dataset.py")


def load_dataset_class():
    """按文件路径加载 Dataset 类，避免仓库内其他 utils 包遮蔽。"""
    spec = importlib.util.spec_from_file_location("pushcube_dataset", DATASET_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.PushCubeFrameDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/wzh/robot/datasets/vla_pushcube_25/preprocessed")
    args = ap.parse_args()
    root = args.root

    checks = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        checks.append((name, status, detail))
        print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))

    # 文件齐全
    need = ["train.npz", "val.npz", "norm_stats.json", "splits.json"]
    check("产出文件齐全", all(os.path.exists(os.path.join(root, f)) for f in need))

    tr = np.load(os.path.join(root, "train.npz"), allow_pickle=True)
    va = np.load(os.path.join(root, "val.npz"), allow_pickle=True)
    ns = json.load(open(os.path.join(root, "norm_stats.json"), encoding="utf-8"))
    sp = json.load(open(os.path.join(root, "splits.json"), encoding="utf-8"))

    fields = ["images", "actions", "instructions", "episode_ids",
              "frame_indices", "task_indices", "categories", "template_ids"]
    check("npz 字段完整", all(f in tr.files and f in va.files for f in fields),
          f"train={len(tr.files)} fields, val={len(va.files)} fields")

    # 图像 / 动作形状与范围
    for tag, d in (("train", tr), ("val", va)):
        im, ac = d["images"], d["actions"]
        check(f"{tag} 图像形状", im.shape[1:] == (3, 224, 224) and im.dtype == np.float32,
              f"{im.shape}, {im.dtype}")
        check(f"{tag} 图像范围[0,1]", 0.0 <= float(im.min()) and float(im.max()) <= 1.0,
              f"[{im.min():.4f}, {im.max():.4f}]")
        check(f"{tag} 动作形状", ac.shape[1] == 4, f"{ac.shape}")
        check(f"{tag} 动作范围[-1,1]", -1.0001 <= float(ac.min()) and float(ac.max()) <= 1.0001,
              f"[{ac.min():.4f}, {ac.max():.4f}]")

    # 划分：20/5、无重叠、类别覆盖
    tr_eps = set(int(e) for e in tr["episode_ids"])
    va_eps = set(int(e) for e in va["episode_ids"])
    check("划分条数 20/5", len(tr_eps) == 20 and len(va_eps) == 5,
          f"train={len(tr_eps)}, val={len(va_eps)}")
    check("训练/验证无重叠", tr_eps.isdisjoint(va_eps))
    check("合并覆盖 25 条", len(tr_eps | va_eps) == 25)
    check("类别分层：验证覆盖全部 5 类", len(set(va["categories"])) == 5,
          f"{dict(Counter(va['categories']))}")
    check("类别分层：训练覆盖全部 5 类", len(set(tr["categories"])) == 5,
          f"{dict(Counter(tr['categories']))}")
    check("splits.json 与 npz 一致",
          set(sp["train_episode_ids"]) == tr_eps and set(sp["val_episode_ids"]) == va_eps)

    # 回合级一致性：同一 episode 内所有帧指令相同
    by_ep = defaultdict(set)
    for e, ins in zip(tr["episode_ids"], tr["instructions"]):
        by_ep[int(e)].add(str(ins))
    check("回合级指令帧内一致", all(len(v) == 1 for v in by_ep.values()))

    # 归一化可逆：反归一化后还原原始物理动作
    PushCubeDataset = load_dataset_class()
    ds = PushCubeDataset(os.path.join(root, "train.npz"),
                         os.path.join(root, "norm_stats.json"))
    raw_src = np.load("/data/wzh/robot/datasets/vla_pushcube_25/triplets/triplets.npz",
                      allow_pickle=True)["actions"]
    starts = np.concatenate([[0], np.cumsum(
        np.load("/data/wzh/robot/datasets/vla_pushcube_25/triplets/triplets.npz",
                allow_pickle=True)["episode_lengths"])])
    # 取 train 前 1 条轨迹的全部帧做还原对比
    e0 = int(tr["episode_ids"][0])
    orig = raw_src[starts[e0]:starts[e0 + 1]]
    mask = np.where(tr["episode_ids"] == e0)[0]
    rec = ds.unnormalize_actions(tr["actions"][mask])
    err = float(np.max(np.abs(rec - orig)))
    check("归一化可逆（反归一化误差<1e-6）", err < 1e-6, f"max_err={err:.2e}")

    # 常量维 gripper 全 0
    check("gripper 常量维标记", ns["action"]["constant_dims"] == [3],
          f"{ns['action']['constant_dims']}")

    # Dataset 单帧读取
    item = ds[0]
    check("Dataset 单帧字段完整",
          all(k in item for k in ("image", "action", "instruction", "prompt",
                                  "image_dino", "image_siglip", "episode_id")),
          f"keys={sorted(item.keys())}")
    check("Dataset 图像张量形状", tuple(item["image"].shape) == (3, 224, 224))
    check("Dataset 动作张量形状", tuple(item["action"].shape) == (4,))
    check("Dataset prompt 模板正确",
          str(item["prompt"]).startswith("In: What action should the robot take to "))
    check("Dataset 双视图派生",
          tuple(item["image_dino"].shape) == (3, 224, 224)
          and tuple(item["image_siglip"].shape) == (3, 224, 224)
          and not torch.equal(item["image_dino"], item["image_siglip"]))

    # DataLoader 批加载
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    check("DataLoader 批加载 image", tuple(batch["image"].shape) == (8, 3, 224, 224),
          f"{tuple(batch['image'].shape)}")
    check("DataLoader 批加载 action", tuple(batch["action"].shape) == (8, 4))
    check("DataLoader 批加载 dino 视图", tuple(batch["image_dino"].shape) == (8, 3, 224, 224))
    check("DataLoader prompt 为列表", isinstance(batch["prompt"], list) and len(batch["prompt"]) == 8)

    total = len(checks)
    passed = sum(1 for _, s, _ in checks if s == "PASS")
    print(f"\n结果: {passed}/{total} 通过")
    out = {
        "status": "PASS" if passed == total else "FAIL",
        "checks": [{"name": n, "status": s, "detail": d} for n, s, d in checks],
        "train_episodes": len(tr_eps),
        "val_episodes": len(va_eps),
        "train_frames": int(len(tr["images"])),
        "val_frames": int(len(va["images"])),
        "action_min": ns["action"]["min"],
        "action_max": ns["action"]["max"],
        "val_category_dist": dict(Counter(va["categories"])),
    }
    with open(os.path.join(root, "verify_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
