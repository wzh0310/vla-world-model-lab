"""OpenVLA LoRA 微调骨架。

模块职责：
- config  : 超参数配置（YAML / 命令行覆盖）
- data    : 动作 token 化、批次整理（4D -> 7D 对齐）
- model   : 模型加载、LoRA 挂载、本项目 unnorm_key 注册
- trainer : 训练循环、验证评估、adapter 保存

所有实现面向容器 `vla` 环境（transformers==4.40.1 / peft==0.11.1 / torch 2.6+cu124）。
"""

from .config import LoRAFinetuneConfig
from .data import ActionTokenizer, FinetuneCollator, expand_action_to_7d, IGNORE_INDEX
from .model import (
    load_model_and_processor,
    prepare_for_lora,
    register_custom_norm_stats,
    count_trainable_params,
)
from .trainer import LoRATrainer

__all__ = [
    "LoRAFinetuneConfig",
    "ActionTokenizer",
    "FinetuneCollator",
    "expand_action_to_7d",
    "IGNORE_INDEX",
    "load_model_and_processor",
    "prepare_for_lora",
    "register_custom_norm_stats",
    "count_trainable_params",
    "LoRATrainer",
]
