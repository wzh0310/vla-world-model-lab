# -*- coding: utf-8 -*-
"""VLA 微调统一工程门面（vla_finetune）。

结构：
    configs/      训练 / LoRA / 运行配置
    data/         数据处理与格式转换脚本
    train.py      微调主入口（门面）
    eval.py       批量评估与泛化测试（门面）
    inference.py  单样本推理（门面）
    run.py        配置驱动一键启动
    utils/        动作归一化 / 可视化 / 运行时工具

底层训练与评估实现位于 src/vla_finetune/ 与 vla/，本包以子进程方式调用，
保持历史实验结果可复现。
"""

__version__ = "2.0.0"
