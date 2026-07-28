"""SFT 训练模块测试 —— sft_loss_fn 与 _resize_embedding。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from classic_chinese_llm.config.settings import ModelConfig
from classic_chinese_llm.model.transformer import TransformerLM
from classic_chinese_llm.training.sft import _resize_embedding, sft_loss_fn


class _MockModel(nn.Module):
    """返回预设 logits 的 mock 模型。"""

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


class TestSFTLossFn:
    """sft_loss_fn 单元测试。"""

    def test_returns_scalar(self) -> None:
        """sft_loss_fn 返回零维标量张量。"""
        batch_size, seq_len, vocab_size = 2, 5, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        model = _MockModel(logits)
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": labels,
        }

        loss = sft_loss_fn(model, batch)

        assert loss.dim() == 0
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_ignore_index_positions_are_masked(self) -> None:
        """被 -100 标记的位置不参与 loss 计算。"""
        batch_size, seq_len, vocab_size = 1, 5, 10
        labels = torch.tensor([[3, 7, -100, -100, -100]])

        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        logits[0, 0, 3] = 1e9
        logits[0, 1, 7] = 1e9
        # 忽略位置给错误高分 — 不影响 loss
        logits[0, 2, 0] = 1e9
        logits[0, 3, 0] = 1e9
        logits[0, 4, 0] = 1e9

        model = _MockModel(logits)
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels,
        }

        loss = sft_loss_fn(model, batch)
        assert loss.item() < 0.01

    def test_all_ignored_returns_nan(self) -> None:
        """全部位置被忽略时，CrossEntropy 分母为 0 返回 nan。"""
        batch_size, seq_len, vocab_size = 2, 5, 10
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        model = _MockModel(logits)
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": labels,
        }

        loss = sft_loss_fn(model, batch)
        assert torch.isnan(loss)

    def test_mixed_ignored_and_active(self) -> None:
        """混合标签：仅活跃位置影响 loss。"""
        batch_size, seq_len, vocab_size = 2, 6, 10
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, 3:] = -100

        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        for b in range(batch_size):
            for s in range(3):
                logits[b, s, labels[b, s]] = 1e9
        logits[:, 3:, :] = torch.randn(batch_size, 3, vocab_size)

        model = _MockModel(logits)
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels,
        }

        loss = sft_loss_fn(model, batch)
        assert loss.item() < 0.01

    def test_loss_decreases_with_better_logits(self) -> None:
        """更好的 logits 对应更低的 loss。"""
        batch_size, seq_len, vocab_size = 2, 4, 10
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits_random = torch.randn(batch_size, seq_len, vocab_size)
        loss_random = sft_loss_fn(
            _MockModel(logits_random),
            {
                "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
                "labels": labels,
            },
        )

        logits_better = logits_random.clone()
        for b in range(batch_size):
            for s in range(seq_len):
                logits_better[b, s, labels[b, s]] += 5.0
        loss_better = sft_loss_fn(
            _MockModel(logits_better),
            {
                "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
                "labels": labels,
            },
        )

        assert loss_better.item() < loss_random.item()


class TestResizeEmbedding:
    """_resize_embedding 单元测试。"""

    @pytest.fixture
    def tiny_config(self) -> ModelConfig:
        """创建小型模型配置（vocab_size=1000）。"""
        return ModelConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )

    def _make_pretrained_weights(
        self, config: ModelConfig
    ) -> tuple[TransformerLM, dict[str, torch.Tensor]]:
        """创建预训练模型和其 state_dict 副本。"""
        model = TransformerLM(config)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        return model, weights

    def test_expand_vocab_size(self, tiny_config: ModelConfig) -> None:
        """扩展 vocab_size：旧 token 权重保留，新 token 随机初始化。"""
        old_vocab_size = tiny_config.vocab_size  # 1000
        new_vocab_size = 1500

        pretrained_model, pretrained_weights = self._make_pretrained_weights(tiny_config)
        old_embed = pretrained_weights["token_embedding.weight"].clone()

        new_config = ModelConfig(
            vocab_size=new_vocab_size,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
        new_model = TransformerLM(new_config)

        result = _resize_embedding(pretrained_weights, new_model, old_vocab_size, new_vocab_size)

        new_embed = result["token_embedding.weight"]
        assert new_embed.shape[0] == new_vocab_size
        assert torch.equal(new_embed[:old_vocab_size], old_embed)

    def test_shrink_vocab_size(self) -> None:
        """缩小 vocab_size：仅保留前 new_vocab_size 个 token 权重。"""
        old_vocab_size = 2000
        new_vocab_size = 1000

        old_config = ModelConfig(
            vocab_size=old_vocab_size,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
        pretrained_model = TransformerLM(old_config)
        pretrained_weights = {k: v.clone() for k, v in pretrained_model.state_dict().items()}
        old_embed = pretrained_weights["token_embedding.weight"].clone()

        new_config = ModelConfig(
            vocab_size=new_vocab_size,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
        new_model = TransformerLM(new_config)

        result = _resize_embedding(pretrained_weights, new_model, old_vocab_size, new_vocab_size)

        new_embed = result["token_embedding.weight"]
        assert new_embed.shape[0] == new_vocab_size
        assert torch.equal(new_embed, old_embed[:new_vocab_size])

    def test_same_vocab_size_no_change(self, tiny_config: ModelConfig) -> None:
        """相同 vocab_size 时权重不被修改。"""
        pretrained_model, pretrained_weights = self._make_pretrained_weights(tiny_config)
        old_embed = pretrained_weights["token_embedding.weight"].clone()

        new_model = TransformerLM(tiny_config)

        result = _resize_embedding(pretrained_weights, new_model, 1000, 1000)

        assert torch.equal(result["token_embedding.weight"], old_embed)

    def test_lm_head_resized_same_as_embedding(self, tiny_config: ModelConfig) -> None:
        """lm_head 与 token_embedding 同时被 resize（tied weights）。"""
        old_vocab_size = tiny_config.vocab_size  # 1000
        new_vocab_size = 2000

        pretrained_model, pretrained_weights = self._make_pretrained_weights(tiny_config)
        old_embed = pretrained_weights["token_embedding.weight"].clone()

        new_config = ModelConfig(
            vocab_size=new_vocab_size,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
        new_model = TransformerLM(new_config)

        result = _resize_embedding(pretrained_weights, new_model, old_vocab_size, new_vocab_size)

        assert "token_embedding.weight" in result
        assert "lm_head.weight" in result
        assert result["token_embedding.weight"].shape[0] == new_vocab_size
        assert result["lm_head.weight"].shape[0] == new_vocab_size
        assert torch.equal(
            result["token_embedding.weight"][:old_vocab_size],
            old_embed[:old_vocab_size],
        )
        assert torch.equal(
            result["token_embedding.weight"][:old_vocab_size],
            result["lm_head.weight"][:old_vocab_size],
        )

    def test_non_embedding_weights_unchanged(self, tiny_config: ModelConfig) -> None:
        """非 embedding/lm_head 的权重保持不变。"""
        old_vocab_size = tiny_config.vocab_size  # 1000
        new_vocab_size = 1500

        pretrained_model, pretrained_weights = self._make_pretrained_weights(tiny_config)

        other_keys = [
            k for k in pretrained_weights if k not in ("token_embedding.weight", "lm_head.weight")
        ]
        other_snapshots = {k: pretrained_weights[k].clone() for k in other_keys}

        new_config = ModelConfig(
            vocab_size=new_vocab_size,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
        new_model = TransformerLM(new_config)

        result = _resize_embedding(pretrained_weights, new_model, old_vocab_size, new_vocab_size)

        for k in other_keys:
            assert torch.equal(result[k], other_snapshots[k]), f"非 embedding 权重 {k} 被意外修改"
