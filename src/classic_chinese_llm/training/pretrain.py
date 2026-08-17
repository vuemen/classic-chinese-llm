"""预训练 (Causal LM Pretraining) —— 全序列 next-token prediction。

数据来自清洗后的文言文 JSONL, 对全序列计算 CrossEntropy loss。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from classic_chinese_llm.config.paths import PathConfig
from classic_chinese_llm.config.settings import PretrainConfig
from classic_chinese_llm.model.transformer import TransformerLM
from classic_chinese_llm.training.callbacks import CheckpointCallback, LoggingCallback
from classic_chinese_llm.training.data_collator import DataCollator
from classic_chinese_llm.training.datasets import PretrainDataset
from classic_chinese_llm.training.trainer import Trainer
from classic_chinese_llm.utils.device import detect_device
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


def pretrain_loss_fn(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Causal Language Modeling 的 loss 计算。

    标准 next-token prediction:
    - 模型通过 causal attention 自动限制每个位置只看向历史 token
    - 位置 t 的 logits 预测位置 t+1 的 token，因此 labels 需右移一位
    - labels 中 PAD 位置已设为 -100 (CrossEntropyLoss 的 ignore_index)

    Args:
        model: TransformerLM 模型。
        batch: {"input_ids": (B, S), "labels": (B, S)}。

    Returns:
        标量 loss。
    """
    logits = model(input_ids=batch["input_ids"])  # (B, S, vocab_size)
    # Causal LM 的 shift: 位置 t 预测 t+1, 丢弃最后一个位置的预测
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


class PretrainRunner:
    """预训练流程编排。

    职责:
    1. 构建预训练数据集 + DataLoader
    2. 实例化模型
    3. 创建 Trainer
    4. 启动训练
    """

    def __init__(
        self,
        config: PretrainConfig,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        self.config = config
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.device_info = detect_device()

    def run(self) -> None:
        """执行完整的预训练流程。"""
        logger.info(
            "开始预训练: d_model=%d, n_layers=%d, n_heads=%d",
            self.config.model.d_model,
            self.config.model.n_layers,
            self.config.model.n_heads,
        )

        # 1. 构建数据集
        train_dataset = PretrainDataset(
            self.data_path,
            self.tokenizer,
            max_seq_len=self.config.model.max_seq_len,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=DataCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=self.config.model.max_seq_len,
                is_sft=False,
            ),
            # Windows 下 DataLoader 用 spawn 启动 worker，会把整个 in-memory
            # dataset（约 20 亿字）pickle 到每个进程，触发 MemoryError。
            # tokenize 在 __getitem__ 里做、开销相对 GPU 可忽略，故用 0 单进程加载。
            num_workers=0,
            pin_memory=True,
        )

        # 2. 创建模型
        model = TransformerLM(self.config.model)
        model.to(self.device_info.device)
        logger.info("模型参数量: %d", model.get_num_params())

        # 3. 创建 Trainer
        paths = PathConfig.get()
        trainer = Trainer(
            model=model,
            config=self.config,
            train_dataloader=train_loader,
            val_dataloader=None,  # 预训练阶段验证集可选
            device_info=self.device_info,
            checkpoint_dir=paths.checkpoint_dir,
            callbacks=[
                LoggingCallback(log_dir=paths.logs_dir),
                CheckpointCallback(),
            ],
            resume=True,
        )

        # 4. 开始训练
        trainer.train(loss_fn=pretrain_loss_fn)
