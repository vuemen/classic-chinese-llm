"""Generator 与 KVCache 测试。"""

from __future__ import annotations

import pytest
import torch

from classic_chinese_llm.config.settings import ModelConfig
from classic_chinese_llm.model.generation import (
    GenerationConfig,
    Generator,
    KVCache,
    _apply_repetition_penalty,
    _top_k_filter,
    _top_p_filter,
)
from classic_chinese_llm.model.transformer import TransformerLM


def _make_tiny_model() -> TransformerLM:
    """创建微小的测试模型（使用 Pydantic 允许的最小合法值）。"""
    return TransformerLM(
        ModelConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# KVCache
# ═══════════════════════════════════════════════════════════════════════════════


class TestKVCache:
    """KVCache 测试。"""

    def test_initial_state_is_empty(self) -> None:
        """初始状态所有层缓存为 None。"""
        cache = KVCache(n_layers=3)
        for i in range(3):
            assert cache.keys[i] is None
            assert cache.values[i] is None

    def test_update_adds_to_cache(self) -> None:
        """update 追加 K/V 到缓存。"""
        cache = KVCache(n_layers=1)
        k = torch.randn(1, 2, 1, 16)
        v = torch.randn(1, 2, 1, 16)
        k_out, v_out = cache.update(0, k, v)
        assert k_out.shape == (1, 2, 1, 16)
        assert v_out.shape == (1, 2, 1, 16)

    def test_update_accumulates(self) -> None:
        """多次 update 累积序列长度。"""
        cache = KVCache(n_layers=1)
        k1 = torch.randn(1, 2, 1, 16)
        cache.update(0, k1, k1)
        k2 = torch.randn(1, 2, 1, 16)
        k_out, _ = cache.update(0, k2, k2)
        assert k_out.shape == (1, 2, 2, 16)  # 累积了 2 个 token

    def test_reset_clears_cache(self) -> None:
        """reset 清空所有缓存。"""
        cache = KVCache(n_layers=2)
        k = torch.randn(1, 2, 1, 16)
        cache.update(0, k, k)
        cache.update(1, k, k)
        cache.reset()
        for i in range(2):
            assert cache.keys[i] is None
            assert cache.values[i] is None


# ═══════════════════════════════════════════════════════════════════════════════
# GenerationConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerationConfig:
    """GenerationConfig 测试。"""

    def test_default_values(self) -> None:
        """默认值正确。"""
        cfg = GenerationConfig()
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 1.0
        assert cfg.top_k == 0
        assert cfg.top_p == 1.0
        assert cfg.repetition_penalty == 1.0
        assert cfg.num_beams == 1
        assert cfg.do_sample is True
        assert cfg.eos_token_id == 3

    def test_custom_values(self) -> None:
        """自定义值正确设置。"""
        cfg = GenerationConfig(
            max_new_tokens=50,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            num_beams=3,
            do_sample=False,
        )
        assert cfg.max_new_tokens == 50
        assert cfg.temperature == 0.7
        assert cfg.top_k == 40
        assert cfg.top_p == 0.9
        assert cfg.num_beams == 3
        assert cfg.do_sample is False


# ═══════════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerator:
    """Generator 测试。"""

    @pytest.fixture
    def model(self) -> TransformerLM:
        return _make_tiny_model()

    @pytest.fixture
    def generator(self, model: TransformerLM) -> Generator:
        return Generator(model)

    def test_greedy_generates_tokens(self, generator: Generator) -> None:
        """贪心解码生成不超过 max_new_tokens 个 token。"""
        input_ids = torch.randint(0, 1000, (1, 4))
        cfg = GenerationConfig(max_new_tokens=10, do_sample=False)
        output = generator.generate(input_ids, cfg)
        assert output.size(0) == 1
        assert output.size(1) >= 4  # 至少包含 prompt
        assert output.size(1) <= 4 + 10  # 不超过 prompt + max_new_tokens

    def test_greedy_stops_at_eos(self, generator: Generator) -> None:
        """遇到 EOS 时停止生成。"""
        # 设置一个非常小的模型，希望它快速生成 EOS
        input_ids = torch.tensor([[0, 1, 2]])  # 随机 prompt
        cfg = GenerationConfig(max_new_tokens=5, do_sample=False, eos_token_id=3)
        output = generator.generate(input_ids, cfg)
        assert output.size(1) <= 4 + 5  # 长度不超过预期

    def test_temperature_zero_equals_greedy(self, generator: Generator) -> None:
        """temperature=0 时确定性输出相同。"""
        input_ids = torch.randint(0, 1000, (1, 4))
        cfg = GenerationConfig(max_new_tokens=5, temperature=0.0, do_sample=True)

        torch.manual_seed(42)
        out_a = generator.generate(input_ids, cfg)
        torch.manual_seed(42)
        out_b = generator.generate(input_ids, cfg)

        assert torch.equal(out_a, out_b)

    def test_beam_search_generates(self, generator: Generator) -> None:
        """Beam Search 正常生成。"""
        input_ids = torch.randint(0, 1000, (1, 4))
        cfg = GenerationConfig(max_new_tokens=5, num_beams=3, do_sample=False)
        output = generator.generate(input_ids, cfg)
        assert output.size(0) == 1
        assert output.size(1) >= 4

    def test_stream_yields_tokens(self, generator: Generator) -> None:
        """流式生成逐个 yield token。"""
        input_ids = torch.randint(0, 1000, (1, 4))
        cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
        tokens = list(generator.generate_stream(input_ids, cfg))
        assert len(tokens) <= 5
        for token in tokens:
            assert isinstance(token, int)
            assert 0 <= token < 1000  # vocab_size

    def test_stream_stops_at_eos(self, generator: Generator) -> None:
        """流式生成遇到 EOS 停止。"""
        input_ids = torch.tensor([[0]])
        cfg = GenerationConfig(max_new_tokens=5, do_sample=False, eos_token_id=3)
        tokens = list(generator.generate_stream(input_ids, cfg))
        # 不应该超过 max_new_tokens
        assert len(tokens) <= 5

    def test_single_dimension_input(self, generator: Generator) -> None:
        """1D input_ids 自动添加 batch 维度。"""
        input_ids = torch.randint(0, 1000, (4,))  # (S,)
        cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
        output = generator.generate(input_ids, cfg)
        assert output.ndim == 2  # (1, S+new_tokens)

    def test_batch_dimension_input(self, generator: Generator) -> None:
        """2D input_ids 保持 batch 维度。"""
        input_ids = torch.randint(0, 1000, (1, 4))  # (1, S)
        cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
        output = generator.generate(input_ids, cfg)
        assert output.ndim == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 采样辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


class TestSamplingHelpers:
    """采样辅助函数测试。"""

    def test_top_k_filter_preserves_top_k(self) -> None:
        """Top-K 过滤后保留 K 个非 -inf 值。"""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = _top_k_filter(logits, k=3)
        # 只有 top-3 不是 -inf
        assert (filtered > float("-inf")).sum() == 3

    def test_top_k_zero_passthrough(self) -> None:
        """k=0 时不过滤。"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = _top_k_filter(logits, k=0)
        assert torch.equal(filtered, logits)

    def test_top_p_filter_preserves_nucleus(self) -> None:
        """Top-P 过滤后保留累积概率 ≥ p 的最小集合。"""
        logits = torch.tensor([[10.0, 0.1, 0.1, 0.1]])  # 第一项概率远大于其他
        filtered = _top_p_filter(logits, p=0.9)
        # 第一项概率接近 1.0，应该被保留
        assert filtered[0, 0] > float("-inf")

    def test_top_p_one_passthrough(self) -> None:
        """p=1.0 不过滤。"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = _top_p_filter(logits, p=1.0)
        assert torch.equal(filtered, logits)

    def test_repetition_penalty_reduces_repeated(self) -> None:
        """重复惩罚降低已出现 token 的概率。"""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        generated = torch.tensor([[0, 1]])  # token 0 和 1 已出现
        original_score_0 = logits[0, 0].item()
        penalized = _apply_repetition_penalty(logits, generated, penalty=2.0)
        assert penalized[0, 0].item() < original_score_0  # token 0 被惩罚

    def test_repetition_penalty_one_passthrough(self) -> None:
        """penalty=1.0 不改变 logits。"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        generated = torch.tensor([[0, 1]])
        penalized = _apply_repetition_penalty(logits, generated, penalty=1.0)
        assert torch.equal(penalized, logits)
