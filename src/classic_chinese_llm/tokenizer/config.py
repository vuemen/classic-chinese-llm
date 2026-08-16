"""Tokenizer 配置模型。

定义 SentencePiece Unigram 训练的全部可配置参数。
所有参数映射到 SentencePieceTrainer.train() 的参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TokenizerConfig:
    """SentencePiece Unigram 训练配置。

    所有参数均有合理默认值，可直接用于文言文 tokenizer 训练。

    用法:
        config = TokenizerConfig(vocab_size=32000)
        trainer = TokenizerTrainer(config)
        trainer.train()
    """

    # ─── 核心参数 ─────────────────────────────────────────────────────

    vocab_size: int = 32000
    """词汇量大小。32K 在压缩率与嵌入参数间取得平衡。"""

    model_type: str = "unigram"
    """分词模型类型: unigram | bpe | char | word。"""

    character_coverage: float = 0.99995
    """字符覆盖率。0.99995 覆盖 ~15,000 个汉字，剩余由 byte_fallback 处理。"""

    byte_fallback: bool = True
    """是否启用 byte fallback。True = 零 OOV 保证。"""

    # ─── 训练参数 ─────────────────────────────────────────────────────

    input_sentence_size: int = 20_000_000
    """训练时采样的最大句读片段数。

    语料在 prepare_corpus 阶段已按句读标点断句（一行 = 一个句读片段，约 12-15 字）。
    全量约 1 亿片段，采样 20M（约 2.4-3 亿字符）在训练速度与 vocab 质量间取得平衡，
    预计 1-2 小时内完成训练。
    """

    shuffle_input_sentence: bool = True
    """是否打乱训练数据。防止语料顺序导致的分布偏差。"""

    num_threads: int = 16
    """训练时并行处理的线程数。"""

    num_sub_iterations: int = 2
    """EM 优化迭代次数。2-3 次是 sweet spot。"""

    # ─── 分词行为参数 ─────────────────────────────────────────────────

    max_sentencepiece_length: int = 16
    """子词最大字符数。16 字足够覆盖文言文固定搭配。"""

    split_by_unicode_script: bool = True
    """是否按 Unicode 区块分片。防止跨脚本不合理合并。"""

    split_by_number: bool = True
    """是否独立处理数字。"""

    split_by_whitespace: bool = True
    """是否在空白字符处分割。"""

    treat_whitespace_as_suffix: bool = False
    """是否将空白视为前一个 token 的后缀。"""

    hard_vocab_limit: bool = True
    """是否强制 vocab_size 精确匹配。True=精确；False=允许实际 vocab 小于设定值。"""

    # ─── 特殊 Token ───────────────────────────────────────────────────

    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    bos_token: str = "<|bos|>"
    eos_token: str = "<|eos|>"

    # ChatML 特殊 token (预训练中用作分隔符, SFT 中用于 Chat Template)
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"

    # ─── 路径 ─────────────────────────────────────────────────────────

    corpus_path: str = "data/processed/deduplicated.jsonl"
    """训练语料路径。默认为 Phase 2 数据管道的去重输出。"""

    model_prefix: str = "models/tokenizer/classical_chinese"
    """SentencePiece 模型文件前缀（不含扩展名）。"""

    output_dir: str = "models/tokenizer"
    """最终 HF tokenizer 输出目录。"""

    # ─── 特殊 Token ID 分配（固定，不可配置）────────────────────────────

    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    # ─── 派生属性 ─────────────────────────────────────────────────────

    @property
    def special_tokens(self) -> list[str]:
        """按 SentencePiece 注册顺序返回所有特殊 token。"""
        return [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.system_token,
            self.user_token,
            self.assistant_token,
            self.end_token,
        ]

    @property
    def user_defined_symbols(self) -> list[str]:
        """返回作为 user_defined_symbols 的额外特殊 token。

        这些 token 不占用 pre-defined id 位置 (0-3)，
        SentencePiece 将它们作为普通 token 加入 vocab 再提升为特殊 token。
        """
        return [
            self.system_token,
            self.user_token,
            self.assistant_token,
            self.end_token,
        ]

    @property
    def model_path(self) -> Path:
        """SentencePiece .model 文件路径。"""
        return Path(f"{self.model_prefix}.model")

    @property
    def vocab_path(self) -> Path:
        """SentencePiece .vocab 文件路径。"""
        return Path(f"{self.model_prefix}.vocab")
