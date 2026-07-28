"""训练层 —— 通用训练框架 + 预训练 + 指令微调。

主要组件:
- Trainer: 通用训练循环 (梯度累积 + 混合精度 + LR 调度 + Checkpoint)
- PretrainRunner: 预训练流程编排
- SFTRunner: 指令微调流程编排
- Callback / LoggingCallback / CheckpointCallback / EarlyStoppingCallback: 回调系统
- DataCollator: 动态 padding + attention mask + SFT label masking
- PretrainDataset / SFTDataset: 训练数据集
"""

from classic_chinese_llm.training.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)
from classic_chinese_llm.training.data_collator import DataCollator
from classic_chinese_llm.training.datasets import PretrainDataset, SFTDataset
from classic_chinese_llm.training.pretrain import PretrainRunner, pretrain_loss_fn
from classic_chinese_llm.training.sft import SFTRunner, sft_loss_fn
from classic_chinese_llm.training.trainer import Trainer

__all__ = [
    "Trainer",
    "PretrainRunner",
    "SFTRunner",
    "pretrain_loss_fn",
    "sft_loss_fn",
    "Callback",
    "LoggingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "DataCollator",
    "PretrainDataset",
    "SFTDataset",
]
