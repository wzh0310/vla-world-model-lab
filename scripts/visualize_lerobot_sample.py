"""从转换后的 LeRobot Parquet 抽检帧，生成图像对齐核验拼图（交付证据）。"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/data/wzh/robot/datasets/vla_pushcube_25/lerobot"
OUT = "/data/wzh/robot/vla-world-model-lab/results/vla_dataset/lerobot_sample_grid.png"


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "train", "data", "chunk-*", "file-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # 取 4 个不同 episode 的首帧
    eps = sorted(df["episode_index"].unique())[:4]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, ep in zip(axes, eps):
        row = df[df["episode_index"] == ep].iloc[0]
        img = row["observation.images.main"]  # parquet 存的是 dict {'bytes','path'} 或 array
        if isinstance(img, dict):
            import io
            from PIL import Image
            img = np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
        else:
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] in (3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            img = arr
        ax.imshow(img)
        instr = str(row["observation.language_instruction"])
        ax.set_title(f"ep{ep} | {instr[:28]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("LeRobot parquet sampled frames - RGB / instruction alignment", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print("saved", OUT)


if __name__ == "__main__":
    main()
