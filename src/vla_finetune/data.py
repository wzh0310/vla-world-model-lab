"""动作 token 化与批次整理。

关键点：
1. 数据集动作是 4 维归一化到 [-1,1]，OpenVLA 词表动作 token 对应 7 维；
   按 [dx,dy,dz,0,0,0,gripper] 扩维，与 configs/pushcube_action.yaml 一致。
2. 动作 token 编号与 OpenVLAForActionPrediction.predict_action 互逆：
   token_id = vocab_size - 1 - bin_index,  bin_index = argmin|bin_center - a|。
3. 只对动作位算 loss：prompt 位 label 置 IGNORE_INDEX；图像 patch 位由模型
   forward 内部自动置 IGNORE_INDEX（见 modeling_prismatic.py 的 multimodal_labels）。
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IGNORE_INDEX = -100


def expand_action_to_7d(action) -> np.ndarray:
    """4D [dx,dy,dz,gripper] -> 7D [dx,dy,dz,0,0,0,gripper]（归一化域 [-1,1]）。"""
    a = np.asarray(action, dtype=np.float32)
    if a.shape[-1] == 7:
        return a
    if a.shape[-1] != 4:
        raise ValueError(f"期望 4 维或 7 维动作，实际 {a.shape}")
    out = np.zeros(a.shape[:-1] + (7,), dtype=np.float32)
    out[..., :3] = a[..., :3]
    out[..., 6] = a[..., 3]
    return out


class ActionTokenizer:
    """把 [-1,1] 连续动作映射为 OpenVLA 离散动作 token id（仅用动作词表尾部）。"""

    def __init__(self, n_action_bins: int, vocab_size: int, action_dim: int = 7):
        bins = np.linspace(-1.0, 1.0, int(n_action_bins))
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0   # [n_bins-1]
        self.vocab_size = int(vocab_size)
        self.action_dim = int(action_dim)

    def encode(self, actions) -> np.ndarray:
        """[B,D] float -> [B,D] int64 token id。"""
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        if a.shape[-1] != self.action_dim:
            raise ValueError(f"动作维度 {a.shape[-1]} != {self.action_dim}")
        idx = np.abs(self.bin_centers[None, None, :] - a[:, :, None]).argmin(axis=-1)
        idx = np.clip(idx, 0, len(self.bin_centers) - 1)
        return (self.vocab_size - 1 - idx).astype(np.int64)

    def decode(self, token_ids) -> np.ndarray:
        """[B,D] token id -> [B,D] [-1,1] 连续动作（用于校验，非推理主路径）。"""
        ids = np.asarray(token_ids, dtype=np.int64)
        disc = np.clip(self.vocab_size - 1 - ids, 0, len(self.bin_centers) - 1)
        return self.bin_centers[disc]


class FinetuneCollator:
    """把 PushCubeFrameDataset 的单帧样本整理为 OpenVLA forward 可直接消费的 batch。"""

    def __init__(self, processor, action_tokenizer: ActionTokenizer, action_dim: int = 7):
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.action_dim = action_dim

    @staticmethod
    def _to_pil(image_tensor: torch.Tensor) -> Image.Image:
        """[3,224,224] float [0,1] -> PIL RGB（processor 内部再做双主干归一化）。"""
        arr = image_tensor.detach().permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in items]
        images = [self._to_pil(item["image"]) for item in items]

        # processor 输出: input_ids/attention_mask [B,L], pixel_values [B,6,224,224]
        batch = self.processor(text=prompts, images=images, return_tensors="pt")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        actions = np.stack([expand_action_to_7d(item["action"].numpy()) for item in items])
        action_tokens = torch.from_numpy(self.action_tokenizer.encode(actions)).long()  # [B,D]

        new_input_ids = torch.cat([input_ids, action_tokens], dim=1)
        new_attention_mask = torch.cat(
            [attention_mask, torch.ones_like(action_tokens)], dim=1
        )
        labels = torch.full_like(new_input_ids, IGNORE_INDEX)
        labels[:, input_ids.shape[1]:] = action_tokens   # 只对动作位监督

        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "pixel_values": batch["pixel_values"],
            "labels": labels,
        }
