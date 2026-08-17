"""指令微调 (SFT) —— Chat template 格式化 + 仅 assistant token 计算 loss。

从预训练 checkpoint 加载权重, 在指令数据上微调, loss 仅计算在 assistant 回复部分。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from classic_chinese_llm.config.paths import PathConfig
from classic_chinese_llm.config.settings import SFTConfig
from classic_chinese_llm.model.transformer import TransformerLM
from classic_chinese_llm.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)
from classic_chinese_llm.training.data_collator import DataCollator
from classic_chinese_llm.training.datasets import SFTDataset
from classic_chinese_llm.training.trainer import Trainer
from classic_chinese_llm.utils.checkpoint import load_checkpoint
from classic_chinese_llm.utils.device import detect_device
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


def sft_loss_fn(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """SFT loss —— labels 已由 DataCollator/SFTDataset 预处理完成。

    非 assistant 位置已设为 -100, 此函数与 pretrain_loss_fn 结构相同,
    同样需要 Causal LM 的 shift (位置 t 预测 t+1)。
    """
    logits = model(input_ids=batch["input_ids"])
    # Causal LM 的 shift: 位置 t 预测 t+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


class SFTRunner:
    """指令微调流程编排。

    职责:
    1. 从预训练 checkpoint 加载模型权重
    2. 构建 SFT 数据集 + DataLoader
    3. 创建 Trainer
    4. 启动 SFT 训练
    """

    def __init__(
        self,
        config: SFTConfig,
        train_data_path: str | Path,
        val_data_path: str | Path | None,
        pretrained_checkpoint: str | Path,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        self.config = config
        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path) if val_data_path else None
        self.pretrained_checkpoint = Path(pretrained_checkpoint)
        self.tokenizer = tokenizer
        self.device_info = detect_device()

    def run(self) -> None:
        """执行完整的 SFT 流程。"""
        logger.info("开始指令微调: pretrained_checkpoint=%s", self.pretrained_checkpoint)

        # 1. 加载预训练权重
        model = self._load_pretrained_model()

        # 2. 构建数据集
        train_dataset = SFTDataset(
            self.train_data_path,
            self.tokenizer,
            max_seq_len=self.config.model.max_seq_len,
            chat_template=self.config.chat_template,
        )
        val_dataset = (
            SFTDataset(
                self.val_data_path,
                self.tokenizer,
                max_seq_len=self.config.model.max_seq_len,
                chat_template=self.config.chat_template,
            )
            if self.val_data_path
            else None
        )

        # 3. DataLoaders
        collator = DataCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.config.model.max_seq_len,
            is_sft=True,
        )
        # num_workers=0: Windows 下 spawn 会把 dataset pickle 到 worker，
        # 与预训练同理由改用单进程加载，避免内存开销。
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                collate_fn=collator,
                num_workers=0,
                pin_memory=True,
            )
            if val_dataset
            else None
        )

        # 4. 创建 Trainer
        paths = PathConfig.get()
        sft_checkpoint_dir = paths.checkpoint_dir / "sft"
        trainer = Trainer(
            model=model,
            config=self.config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device_info=self.device_info,
            checkpoint_dir=sft_checkpoint_dir,
            callbacks=[
                LoggingCallback(log_dir=paths.logs_dir),
                CheckpointCallback(),
                EarlyStoppingCallback(patience=5),
            ],
            resume=False,  # SFT 从预训练权重开始, 不自动 resume
        )

        # 5. 开始训练
        trainer.train(loss_fn=sft_loss_fn)

    def _load_pretrained_model(self) -> TransformerLM:
        """从预训练 checkpoint 加载模型权重。

        仅加载模型权重 (不加载 optimizer/scheduler 状态),
        SFT 使用全新的 optimizer。

        Returns:
            加载了预训练权重的 TransformerLM 模型。
        """
        logger.info("加载预训练权重: %s", self.pretrained_checkpoint)
        state = load_checkpoint(self.pretrained_checkpoint, map_location="cpu")

        model = TransformerLM(self.config.model)

        # 处理 vocab 大小不匹配
        pretrained_weights = state.model_state_dict
        pretrained_vocab_size = pretrained_weights["token_embedding.weight"].shape[0]
        current_vocab_size = self.config.model.vocab_size

        if pretrained_vocab_size != current_vocab_size:
            logger.warning(
                "vocab_size 不匹配: pretrained=%d, current=%d, 将 resize embedding",
                pretrained_vocab_size,
                current_vocab_size,
            )
            pretrained_weights = _resize_embedding(
                pretrained_weights,
                model,
                pretrained_vocab_size,
                current_vocab_size,
            )

        # strict=False: 允许新添加的特殊 token embedding 随机初始化
        missing, unexpected = model.load_state_dict(pretrained_weights, strict=False)
        if missing:
            logger.warning("缺失的权重 (将随机初始化): %s", missing)
        if unexpected:
            logger.warning("多余的权重 (将忽略): %s", unexpected)

        model.to(self.device_info.device)
        logger.info("预训练权重加载完成: %d 参数", model.get_num_params())
        return model


def _resize_embedding(
    pretrained_weights: dict[str, torch.Tensor],
    model: TransformerLM,
    old_vocab_size: int,
    new_vocab_size: int,
) -> dict[str, torch.Tensor]:
    """调整预训练权重的 embedding 大小。

    当 tokenizer 添加了新的特殊 token 时,
    embedding 矩阵需要扩展, 新 token 的 embedding 随机初始化。

    Args:
        pretrained_weights: 预训练的 state_dict。
        model: 目标模型。
        old_vocab_size: 预训练的 vocab_size。
        new_vocab_size: 当前的 vocab_size。

    Returns:
        调整后的 state_dict。
    """
    for key in ["token_embedding.weight", "lm_head.weight"]:
        if key in pretrained_weights:
            old_weight = pretrained_weights[key]
            if old_weight.shape[0] != new_vocab_size:
                new_weight = model.state_dict()[key].clone()
                # 复制可用的部分
                copy_size = min(old_vocab_size, new_vocab_size)
                new_weight[:copy_size] = old_weight[:copy_size]
                pretrained_weights[key] = new_weight

    return pretrained_weights
