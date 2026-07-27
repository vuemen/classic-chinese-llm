"""TokenizerConfig 测试。"""

from __future__ import annotations

from pathlib import Path

from classic_chinese_llm.tokenizer.config import TokenizerConfig


class TestTokenizerConfig:
    """TokenizerConfig 默认值和属性测试。"""

    def test_default_values(self) -> None:
        """默认值符合设计规格。"""
        cfg = TokenizerConfig()

        assert cfg.vocab_size == 32000
        assert cfg.model_type == "unigram"
        assert cfg.character_coverage == 0.99995
        assert cfg.byte_fallback is True
        assert cfg.input_sentence_size == 10_000_000
        assert cfg.shuffle_input_sentence is True
        assert cfg.num_threads == 16
        assert cfg.num_sub_iterations == 2
        assert cfg.max_sentencepiece_length == 16

    def test_special_token_ids(self) -> None:
        """特殊 token ID 分配符合 SentencePiece 约定。"""
        cfg = TokenizerConfig()

        assert cfg.pad_id == 0
        assert cfg.unk_id == 1
        assert cfg.bos_id == 2
        assert cfg.eos_id == 3

    def test_special_tokens_property(self) -> None:
        """special_tokens 属性包含所有 8 个特殊 token。"""
        cfg = TokenizerConfig()
        tokens = cfg.special_tokens

        assert len(tokens) == 8
        assert tokens[0] == cfg.pad_token
        assert tokens[1] == cfg.unk_token
        assert tokens[2] == cfg.bos_token
        assert tokens[3] == cfg.eos_token
        assert cfg.system_token in tokens
        assert cfg.user_token in tokens
        assert cfg.assistant_token in tokens
        assert cfg.end_token in tokens

    def test_user_defined_symbols(self) -> None:
        """user_defined_symbols 包含 4 个 ChatML token。"""
        cfg = TokenizerConfig()
        symbols = cfg.user_defined_symbols

        assert len(symbols) == 4
        assert cfg.system_token in symbols
        assert cfg.user_token in symbols
        assert cfg.assistant_token in symbols
        assert cfg.end_token in symbols

    def test_model_path_property(self) -> None:
        """model_path 属性正确拼接路径。"""
        cfg = TokenizerConfig(model_prefix="models/tokenizer/classical_chinese")

        assert cfg.model_path == Path("models/tokenizer/classical_chinese.model")
        assert cfg.vocab_path == Path("models/tokenizer/classical_chinese.vocab")

    def test_custom_values(self) -> None:
        """自定义参数正确覆盖默认值。"""
        cfg = TokenizerConfig(
            vocab_size=16000,
            model_type="bpe",
            character_coverage=0.999,
            byte_fallback=False,
            num_threads=8,
        )

        assert cfg.vocab_size == 16000
        assert cfg.model_type == "bpe"
        assert cfg.character_coverage == 0.999
        assert cfg.byte_fallback is False
        assert cfg.num_threads == 8

    def test_output_paths_derived_from_model_prefix(self) -> None:
        """output_dir 和 model_prefix 独立设置。"""
        cfg = TokenizerConfig(
            model_prefix="/tmp/test/cc",
            output_dir="/tmp/output",
        )

        assert cfg.model_path == Path("/tmp/test/cc.model")
        assert cfg.output_dir == "/tmp/output"
