#!/usr/bin/env python3
"""Tokenizer 训练 CLI 入口。

用法:
    python scripts/train_tokenizer.py \\
        --corpus data/processed/deduplicated.jsonl \\
        --vocab-size 32000 \\
        --output-dir models/tokenizer

训练 → 封装 → 保存 全流程。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from classic_chinese_llm.config.paths import PathConfig
from classic_chinese_llm.tokenizer.config import TokenizerConfig
from classic_chinese_llm.tokenizer.trainer import TokenizerTrainer
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer, save_tokenizer
from classic_chinese_llm.utils.logging_config import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="训练文言文 SentencePiece Unigram Tokenizer",
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/deduplicated.jsonl",
        help="训练语料路径（JSONL 格式，需含 text 字段）",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="词汇量大小（默认 32000）",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.99995,
        help="字符覆盖率（默认 0.99995）",
    )
    parser.add_argument(
        "--output-dir",
        default="models/tokenizer",
        help="输出目录（默认 models/tokenizer）",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="训练线程数（默认 16）",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="跳过语料准备（当 --corpus 直接指向 txt 文件时使用）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI 主入口。"""
    args = parse_args(argv)

    # 初始化路径
    project_root = Path(__file__).resolve().parent.parent
    PathConfig.initialize(project_root)
    paths = PathConfig.get()
    setup_logging(level="INFO", log_file=str(paths.logs_dir / "train_tokenizer.log"))
    logger = logging.getLogger(__name__)

    # 构建配置
    config = TokenizerConfig(
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        corpus_path=args.corpus,
        model_prefix=str(Path(args.output_dir) / "classical_chinese"),
        output_dir=args.output_dir,
        num_threads=args.num_threads,
    )

    # Step 1: 训练 SentencePiece 模型
    trainer = TokenizerTrainer(config)
    if args.skip_prepare:
        model_path = trainer.train(corpus_path=Path(args.corpus))
    else:
        model_path = trainer.train()

    # Step 2: 封装为 HF PreTrainedTokenizerFast
    hf_tokenizer = build_tokenizer(model_path, config)

    # Step 3: 保存
    save_tokenizer(hf_tokenizer, args.output_dir)

    # Step 4: 验证
    test_text = "子曰：「學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？」"
    tokens = hf_tokenizer.encode(test_text)
    decoded = hf_tokenizer.decode(tokens)
    logger.info(
        "验证编码/解码: '%s' → %d tokens → '%s'",
        test_text,
        len(tokens),
        decoded,
    )

    logger.info("Tokenizer 训练全流程完成！输出目录: %s", args.output_dir)


if __name__ == "__main__":
    main()
