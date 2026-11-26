"""PyTorch Dataset：按索引读取单帧「图像 - 指令 - 动作」三元组。

数据由 scripts/preprocess_vla_data.py 产出：图像已 resize 224×224、转 RGB/CHW、
像素归一化到 [0, 1]；动作已做数据集级 min-max 归一化到 [-1, 1]。
本 Dataset 在 __getitem__ 时按 OpenVLA preprocessor_config.json 的口径派生
DINOv2 / SigLIP 两个视觉视图，并可选拼出完整 prompt，供微调训练直接消费。
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_PROMPT = "In: What action should the robot take to {instruction}?\nOut:"

# 与 openvla-7b/preprocessor_config.json 的 means / stds 字段严格一致；
# 若模型目录可用，优先从 norm_stats.json 读取（预处理脚本已落盘）。
_FALLBACK_VIEWS = {
    "dino": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "siglip": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
}


class PushCubeFrameDataset(Dataset):
    """逐帧读取预处理后的 PushCube VLA 三元组。

    参数
    ----
    npz_path : str | Path
        train.npz 或 val.npz。
    norm_stats_path : str | Path | None
        norm_stats.json 路径；用于读取双视图 mean/std 与动作反归一化参数。
    split : str
        仅作元信息标记（"train" / "val"）。
    with_prompt : bool
        为真时返回完整 OpenVLA prompt（默认真）。
    return_views : bool
        为真时返回 image_dino / image_siglip 两个模型视图（默认真）。

    返回（__getitem__）
    ------------------
    dict:
        image        : FloatTensor [3,224,224]，RGB，像素 [0,1]
        image_dino   : FloatTensor [3,224,224]，DINOv2 视图（ImageNet 归一化）
        image_siglip : FloatTensor [3,224,224]，SigLIP 视图（[0.5] 归一化）
        action       : FloatTensor [4]，归一化到 [-1,1] 的 [dx,dy,dz,gripper]
        instruction  : str，回合级语言指令（该帧所属轨迹共享）
        prompt       : str，完整 OpenVLA 输入文本
        episode_id   : int
        frame_index  : int，帧在所属轨迹内的序号
        task_index   : int，指令模板映射到的 task id
        category     : str，指令类别
    """

    def __init__(
        self,
        npz_path,
        norm_stats_path=None,
        split="train",
        with_prompt=True,
        return_views=True,
    ):
        self.npz_path = str(npz_path)
        self.split = split
        self.with_prompt = with_prompt
        self.return_views = return_views

        with np.load(self.npz_path, allow_pickle=True) as d:
            self.images = d["images"]              # [N,3,224,224] float32, [0,1]
            self.actions = d["actions"]            # [N,4] float32, [-1,1]
            self.instructions = d["instructions"]   # [N] object(str)
            self.episode_ids = d["episode_ids"]     # [N] int64
            self.frame_indices = d["frame_indices"]  # [N] int64
            self.task_indices = d["task_indices"]    # [N] int64
            self.categories = d["categories"]        # [N] object(str)
            if "template_ids" in d.files:
                self.template_ids = d["template_ids"]
            else:
                self.template_ids = np.array([""] * len(self.images), dtype=object)

        assert self.images.ndim == 4 and self.images.shape[1:] == (3, 224, 224), (
            f"图像形状异常: {self.images.shape}，期望 [N,3,224,224]"
        )
        assert self.actions.ndim == 2 and self.actions.shape[1] == 4, (
            f"动作形状异常: {self.actions.shape}，期望 [N,4]"
        )

        self.norm_stats = None
        if norm_stats_path is not None and Path(norm_stats_path).exists():
            with open(norm_stats_path, encoding="utf-8") as f:
                self.norm_stats = json.load(f)

        views = (
            self.norm_stats["image"]["views"]
            if self.norm_stats and "views" in self.norm_stats.get("image", {})
            else _FALLBACK_VIEWS
        )
        self._view_params = {}
        for name, p in views.items():
            self._view_params[name] = (
                np.asarray(p["mean"], dtype=np.float32).reshape(1, 3, 1, 1),
                np.asarray(p["std"], dtype=np.float32).reshape(1, 3, 1, 1),
            )

    def __len__(self):
        return int(len(self.images))

    def __getitem__(self, idx):
        img = torch.from_numpy(np.ascontiguousarray(self.images[idx]))  # [3,224,224]
        act = torch.from_numpy(np.ascontiguousarray(self.actions[idx]))  # [4]
        instr = str(self.instructions[idx])

        item = {
            "image": img,
            "action": act,
            "instruction": instr,
            "episode_id": int(self.episode_ids[idx]),
            "frame_index": int(self.frame_indices[idx]),
            "task_index": int(self.task_indices[idx]),
            "category": str(self.categories[idx]),
            "template_id": str(self.template_ids[idx]),
        }
        if self.with_prompt:
            item["prompt"] = DEFAULT_PROMPT.format(instruction=instr)
        if self.return_views:
            base = img.unsqueeze(0)  # [1,3,224,224]
            for name, (mean, std) in self._view_params.items():
                m = torch.from_numpy(mean).reshape(1, 3, 1, 1)
                s = torch.from_numpy(std).reshape(1, 3, 1, 1)
                item[f"image_{name}"] = ((base - m) / s).squeeze(0)
        return item

    def unnormalize_actions(self, actions):
        """把 [-1,1] 归一化动作还原为物理增量（复用 norm_stats.json）。"""
        if not self.norm_stats:
            raise ValueError("未加载 norm_stats.json，无法反归一化")
        a = np.asarray(actions, dtype=np.float64)
        info = self.norm_stats["action"]
        lo = np.asarray(info["min"], dtype=np.float64)
        hi = np.asarray(info["max"], dtype=np.float64)
        span = hi - lo
        out = np.empty_like(a)
        for j in range(a.shape[-1]):
            if span[j] > 1e-8:
                out[..., j] = lo[j] + (a[..., j] + 1.0) * span[j] / 2.0
            else:
                out[..., j] = lo[j]  # 常量维（gripper 全 0）
        return out
