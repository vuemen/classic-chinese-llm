"""预训练 loss 函数测试 —— pretrain_loss_fn。"""

from __future__ import annotations

import torch
import torch.nn as nn

from classic_chinese_llm.training.pretrain import pretrain_loss_fn


class _MockModel(nn.Module):
    """返回预设 logits 的 mock 模型，用于隔离测试 loss 函数。"""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        vocab_size = self._logits.size(-1)
        if (
            self._logits.dim() == 3
            and self._logits.size(0) == batch_size
            and self._logits.size(1) == seq_len
        ):
            return self._logits
        return self._logits.expand(batch_size, seq_len, vocab_size)


class TestPretrainLossFn:
    """pretrain_loss_fn 单元测试。"""

    # ── 基础行为 ──────────────────────────────────────────────────────

    def test_returns_scalar(self) -> None:
        """pretrain_loss_fn 返回零维标量张量。"""
        batch_size, seq_len, vocab_size = 2, 5, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        model = _MockModel(logits)
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": labels,
        }

        loss = pretrain_loss_fn(model, batch)

        assert loss.dim() == 0
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    # ── ignore_index 行为 ────────────────────────────────────────────

    def test_ignore_index_positions_are_masked(self) -> None:
        """被 -100 标记的位置不参与 loss 计算。"""
        batch_size, seq_len, vocab_size = 1, 5, 10
        labels = torch.tensor([[3, 7, -100, -100, -100]])

        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        # shift 后: 位置 0 预测 labels[1]=7
        logits[0, 0, 7] = 1e9
        # 忽略位置给错误高分 — 不影响 loss
        logits[0, 1, 0] = 1e9
        logits[0, 2, 0] = 1e9
        logits[0, 3, 0] = 1e9

        model = _MockModel(logits)
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels,
        }

        loss = pretrain_loss_fn(model, batch)
        assert loss.item() < 0.01

    def test_all_ignored_returns_nan(self) -> None:
        """全部位置被 -100 标记时，CrossEntropy 分母为 0 返回 nan。"""
        batch_size, seq_len, vocab_size = 2, 5, 10
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        model = _MockModel(logits)
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": labels,
        }

        loss = pretrain_loss_fn(model, batch)
        assert torch.isnan(loss)

    def test_mixed_ignored_and_active(self) -> None:
        """混合标注：仅活跃位置影响 loss。"""
        batch_size, seq_len, vocab_size = 2, 6, 10
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, 3:] = -100

        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        # shift 后: 位置 s 预测 labels[s+1]。活跃位置 0,1 (labels[1], labels[2] 有效)
        for b in range(batch_size):
            for s in range(2):
                logits[b, s, labels[b, s + 1]] = 1e9
        # 忽略位置 (shift 后位置 2,3,4) 给随机值
        logits[:, 2:, :] = torch.randn(batch_size, seq_len - 2, vocab_size)

        model = _MockModel(logits)
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels,
        }

        loss = pretrain_loss_fn(model, batch)
        assert loss.item() < 0.01

    # ── 预测质量 ──────────────────────────────────────────────────────

    def test_perfect_prediction_gives_low_loss(self) -> None:
        """完美预测时 loss 接近 0。"""
        vocab_size = 10
        batch_size, seq_len = 2, 5
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        # 完美预测: 位置 s 预测 labels[s+1]
        logits[:, :-1] = logits[:, :-1].scatter(-1, labels[:, 1:].unsqueeze(-1), 1e9)

        model = _MockModel(logits)
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels,
        }

        loss = pretrain_loss_fn(model, batch)
        assert loss.item() < 0.01

    def test_loss_decreases_with_better_logits(self) -> None:
        """更好的 logits（向正确标签偏移）产生更低的 loss。"""
        batch_size, seq_len, vocab_size = 2, 4, 10
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits_random = torch.randn(batch_size, seq_len, vocab_size)
        model_random = _MockModel(logits_random)
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels,
        }
        loss_random = pretrain_loss_fn(model_random, batch)

        logits_better = logits_random.clone()
        for b in range(batch_size):
            for s in range(seq_len - 1):
                logits_better[b, s, labels[b, s + 1]] += 5.0
        model_better = _MockModel(logits_better)
        loss_better = pretrain_loss_fn(model_better, batch)

        assert loss_better.item() < loss_random.item()

    def test_random_prediction_gives_higher_loss_than_perfect(self) -> None:
        """随机预测的 loss 严格高于完美预测。"""
        vocab_size = 10
        batch_size, seq_len = 2, 5
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        model_random = _MockModel(torch.randn(batch_size, seq_len, vocab_size))
        loss_random = pretrain_loss_fn(
            model_random,
            {
                "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
                "labels": labels,
            },
        )

        logits_perfect = torch.full((batch_size, seq_len, vocab_size), -1e9)
        logits_perfect[:, :-1] = logits_perfect[:, :-1].scatter(
            -1, labels[:, 1:].unsqueeze(-1), 1e9
        )
        model_perfect = _MockModel(logits_perfect)
        loss_perfect = pretrain_loss_fn(
            model_perfect,
            {
                "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
                "labels": labels,
            },
        )

        assert loss_random.item() > loss_perfect.item()

    # ── 边界情况 ──────────────────────────────────────────────────────

    def test_minimal_two_token_sequence(self) -> None:
        """两 token 序列（最小有效长度）正常工作。"""
        batch_size, seq_len, vocab_size = 1, 2, 10
        labels = torch.tensor([[5, 3]])
        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        logits[0, 0, 3] = 1e9  # 位置 0 预测 labels[1]=3

        model = _MockModel(logits)
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels,
        }

        loss = pretrain_loss_fn(model, batch)
        assert loss.dim() == 0
        assert loss.item() < 0.01

    def test_copying_input_is_not_zero_loss(self) -> None:
        """回归测试: 模型"复制"输入 token 不会让 loss 接近 0（shift 生效）。"""
        vocab_size = 10
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        # 模型在每个位置给"自身 token"最高分（复制行为）
        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        logits = logits.scatter(-1, input_ids.unsqueeze(-1), 1e9)

        model = _MockModel(logits)
        batch = {"input_ids": input_ids, "labels": labels}

        loss = pretrain_loss_fn(model, batch)
        # 正确答案是下一个 token，复制自身应产生高 loss
        assert loss.item() > 1.0

    def test_large_batch(self) -> None:
        """较大 batch size 正常工作。"""
        batch_size, seq_len, vocab_size = 4, 8, 20
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        model = _MockModel(logits)
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": labels,
        }

        loss = pretrain_loss_fn(model, batch)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
