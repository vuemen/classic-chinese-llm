"""SentencePiece 训练封装。

封装 SentencePiece Unigram 模型的训练流程：
1. prepare_corpus: 从 deduplicated.jsonl 提取纯文本
2. train: 调用 SentencePieceTrainer 训练模型
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm

from classic_chinese_llm.tokenizer.config import TokenizerConfig
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


class TokenizerTrainer:
    """封装 SentencePiece Unigram 模型的训练流程。

    使用方式:
        config = TokenizerConfig(vocab_size=32000)
        trainer = TokenizerTrainer(config)
        model_path = trainer.train()  # 自动 prepare_corpus + train
    """

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self._output_dir = Path(config.output_dir)
        self._model_prefix = Path(config.model_prefix)

    def prepare_corpus(self) -> Path:
        """从 deduplicated.jsonl 提取纯文本作为训练语料。

        Returns:
            训练语料 txt 文件的路径。

        Raises:
            FileNotFoundError: 语料文件不存在时抛出。
        """
        corpus_input = Path(self.config.corpus_path)
        corpus_output = self._output_dir / "train_corpus.txt"

        if not corpus_input.exists():
            raise FileNotFoundError(
                f"语料文件不存在: {corpus_input}。" f"请先运行 scripts/collect_data.py 完成数据管道"
            )

        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("正在准备训练语料: %s → %s", corpus_input, corpus_output)
        line_count = 0
        char_count = 0

        with (
            open(corpus_input, encoding="utf-8") as f_in,
            open(corpus_output, "w", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "").strip()
                if text:
                    # 最终清洗：多余空白合并
                    cleaned = " ".join(text.split())
                    f_out.write(cleaned + "\n")
                    line_count += 1
                    char_count += len(cleaned)

        logger.info(
            "语料准备完成: %d 行, %d 字符, 文件 %s",
            line_count,
            char_count,
            corpus_output,
        )
        return corpus_output

    def train(self, corpus_path: Path | None = None) -> Path:
        """训练 SentencePiece Unigram 模型。

        Args:
            corpus_path: 训练语料 txt 路径。若为 None，则自动调用 prepare_corpus()。

        Returns:
            生成的 .model 文件路径。

        Raises:
            FileNotFoundError: 语料文件不存在且 prepare_corpus 失败时抛出。
        """
        if corpus_path is None:
            corpus_path = self.prepare_corpus()

        cfg = self.config
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "开始训练 SentencePiece Unigram 模型: vocab_size=%d, "
            "character_coverage=%.5f, model_prefix=%s",
            cfg.vocab_size,
            cfg.character_coverage,
            cfg.model_prefix,
        )

        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(self._model_prefix),
            vocab_size=cfg.vocab_size,
            model_type=cfg.model_type,
            character_coverage=cfg.character_coverage,
            byte_fallback=cfg.byte_fallback,
            input_sentence_size=cfg.input_sentence_size,
            shuffle_input_sentence=cfg.shuffle_input_sentence,
            num_threads=cfg.num_threads,
            num_sub_iterations=cfg.num_sub_iterations,
            max_sentencepiece_length=cfg.max_sentencepiece_length,
            split_by_unicode_script=cfg.split_by_unicode_script,
            split_by_number=cfg.split_by_number,
            split_by_whitespace=cfg.split_by_whitespace,
            treat_whitespace_as_suffix=cfg.treat_whitespace_as_suffix,
            hard_vocab_limit=cfg.hard_vocab_limit,
            # 特殊 token
            pad_id=cfg.pad_id,
            unk_id=cfg.unk_id,
            bos_id=cfg.bos_id,
            eos_id=cfg.eos_id,
            pad_piece=cfg.pad_token,
            unk_piece=cfg.unk_token,
            bos_piece=cfg.bos_token,
            eos_piece=cfg.eos_token,
            # 额外特殊 token（ChatML）
            user_defined_symbols=cfg.user_defined_symbols,
        )

        model_path = cfg.model_path
        vocab_path = cfg.vocab_path

        logger.info(
            "训练完成: model=%s, vocab=%s",
            model_path,
            vocab_path,
        )
        return model_path
