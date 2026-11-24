#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规范入口：世界模型四损失联合训练。

真正的训练实现位于 ``trainer/train_world_model.py``；本文件只做路径引导，
保证仓库规范中的 ``scripts/train_world_model.py`` 可直接运行。

用法（vla 环境）：
    python scripts/train_world_model.py --config configs/default.yaml --output-dir outputs/wm
    python scripts/train_world_model.py --max-steps 200 --output-dir outputs/smoke
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainer.train_world_model import main  # noqa: E402


if __name__ == "__main__":
    main()
