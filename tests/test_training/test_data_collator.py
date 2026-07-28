"""DataCollator 测试 —— 动态 padding + attention mask + label 构建。"""

from __future__ import annotations

import torch

from classic_chinese_llm.training.data_collator import DataCollator


class TestDataCollator:
    """DataCollator 单元测试。"""

    def test_dynamic_padding_to_batch_max(self) -> None:
        """batch 内 pad 到 batch 内的最大长度, 非全局 max_length。"""
        collator = DataCollator(pad_token_id=0, max_length=100)
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5])},
        ]
        result = collator(batch)
        # batch 内最大长度是 3
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)

    def test_attention_mask_values(self) -> None:
        """attention_mask: 真实 token=1, padding=0。"""
        collator = DataCollator(pad_token_id=0, max_length=100)
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4]), "labels": torch.tensor([4])},
        ]
        result = collator(batch)
        expected_mask = torch.tensor(
            [
                [1, 1, 1],
                [1, 0, 0],
            ]
        )
        assert torch.equal(result["attention_mask"], expected_mask)

    def test_padding_value_is_pad_token_id(self) -> None:
        """padding 位置填充 PAD token ID。"""
        collator = DataCollator(pad_token_id=99, max_length=100)
        batch = [
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2])},
            {"input_ids": torch.tensor([3, 4, 5]), "labels": torch.tensor([3, 4, 5])},
        ]
        result = collator(batch)
        # 第一个样本 pad 了 1 个 token
        assert result["input_ids"][0, 2] == 99

    def test_labels_padding_is_ignore_index(self) -> None:
        """labels 中 padding 位置为 -100 (ignore_index)。"""
        collator = DataCollator(pad_token_id=0, max_length=100)
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4]), "labels": torch.tensor([4])},
        ]
        result = collator(batch)
        assert result["labels"][1, 1] == -100
        assert result["labels"][1, 2] == -100

    def test_truncation_to_max_length(self) -> None:
        """超过 max_length 的样本被截断。"""
        collator = DataCollator(pad_token_id=0, max_length=3)
        batch = [
            {"input_ids": torch.tensor([1, 2, 3, 4, 5]), "labels": torch.tensor([1, 2, 3, 4, 5])},
        ]
        result = collator(batch)
        assert result["input_ids"].size(1) == 3
        assert torch.equal(result["input_ids"][0], torch.tensor([1, 2, 3]))

    def test_single_sample_batch(self) -> None:
        """单个样本的 batch 正常工作。"""
        collator = DataCollator(pad_token_id=0, max_length=100)
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
        ]
        result = collator(batch)
        assert result["input_ids"].shape == (1, 3)

    def test_empty_labels_in_item(self) -> None:
        """样本无 labels 时自动从 input_ids 构建。"""
        collator = DataCollator(pad_token_id=0, max_length=100)
        batch = [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4])},
        ]
        result = collator(batch)
        assert "labels" in result
        # padding 位置为 -100
        assert result["labels"][1, 1] == -100
        assert result["labels"][1, 2] == -100
