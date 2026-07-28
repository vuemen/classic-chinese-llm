"""Data Collator —— 动态 padding + attention mask + SFT label masking。

将 Dataset 返回的变长 token 序列整理为固定长度的 batch tensor。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DataCollator:
    """动态 padding 的批次整理器。

    功能:
    1. 将 batch 中的样本 padding 到 batch 内最大长度 (非全局 max_seq_len)
    2. 构建 attention_mask (1 = 真实 token, 0 = padding)
    3. 构建 labels (pretrain: labels = input_ids; SFT: 非 assistant 位置 = -100)

    Args:
        pad_token_id: PAD token ID。
        max_length: 强制截断的最大长度。
        is_sft: 是否 SFT 模式 (SFT 样本的 labels 已由 SFTDataset 预处理)。
    """

    def __init__(
        self,
        pad_token_id: int,
        max_length: int = 2048,
        is_sft: bool = False,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.is_sft = is_sft

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """整理一个 batch。

        Args:
            batch: [{"input_ids": (S1,), "labels": (S1,)}, ...]

        Returns:
            {"input_ids": (B, max_S), "attention_mask": (B, max_S), "labels": (B, max_S)}
        """
        # 强制截断到 max_length
        for item in batch:
            item["input_ids"] = item["input_ids"][: self.max_length]
            if "labels" in item:
                item["labels"] = item["labels"][: self.max_length]

        # 动态 padding: 取 batch 内最大长度
        batch_max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = batch_max_len - seq_len

            # input_ids: 右侧 padding
            input_ids = F.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id)
            input_ids_list.append(input_ids)

            # attention_mask: 1 = 真实 token, 0 = padding
            attn_mask = torch.cat(
                [
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]
            )
            attention_mask_list.append(attn_mask)

            # labels: 同 input_ids, padding 位置设为 -100 (ignore_index)
            if "labels" in item:
                labels = F.pad(item["labels"], (0, pad_len), value=-100)
            else:
                labels = input_ids.clone()
                labels[seq_len:] = -100
            labels_list.append(labels)

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
