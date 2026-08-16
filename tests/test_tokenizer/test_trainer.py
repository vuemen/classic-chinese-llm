"""TokenizerTrainer 测试。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from classic_chinese_llm.tokenizer.config import TokenizerConfig
from classic_chinese_llm.tokenizer.trainer import TokenizerTrainer


def _write_corpus(path: Path, docs: list[str]) -> None:
    """写入一个 JSONL 语料文件，每条记录含 text 字段。"""
    corpus = path / "deduplicated.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        for text in docs:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def _make_trainer(tmp_path: Path, input_sentence_size: int = 20_000_000) -> TokenizerTrainer:
    """构造指向临时目录的 TokenizerTrainer。"""
    config = TokenizerConfig(
        corpus_path=str(tmp_path / "deduplicated.jsonl"),
        output_dir=str(tmp_path / "out"),
        model_prefix=str(tmp_path / "out" / "classical_chinese"),
        input_sentence_size=input_sentence_size,
    )
    return TokenizerTrainer(config)


class TestTokenizerTrainer:
    """TokenizerTrainer 单元测试。"""

    def test_prepare_corpus_splits_documents_into_sentences(self, tmp_path: Path) -> None:
        """每个文档按句读标点断句，一行 = 一个片段。"""
        _write_corpus(tmp_path, ["子曰，學而時習之。不亦說乎？"])
        trainer = _make_trainer(tmp_path)

        out = trainer.prepare_corpus()

        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert lines == ["子曰，", "學而時習之。", "不亦說乎？"]

    def test_prepare_corpus_handles_multiple_records(self, tmp_path: Path) -> None:
        """多个文档逐条断句，顺序保持。"""
        _write_corpus(tmp_path, ["學而時習之。", "不亦樂乎。"])
        trainer = _make_trainer(tmp_path)

        out = trainer.prepare_corpus()

        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert lines == ["學而時習之。", "不亦樂乎。"]

    def test_prepare_corpus_skips_empty_text(self, tmp_path: Path) -> None:
        """空 text 或纯空白记录被跳过。"""
        _write_corpus(tmp_path, ["", "   ", "子曰。學之。"])
        trainer = _make_trainer(tmp_path)

        out = trainer.prepare_corpus()

        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert lines == ["子曰。", "學之。"]

    def test_prepare_corpus_no_punctuation_keeps_single_line(self, tmp_path: Path) -> None:
        """无标点文档退化为单行。"""
        _write_corpus(tmp_path, ["子曰學而時習之不亦說乎"])
        trainer = _make_trainer(tmp_path)

        out = trainer.prepare_corpus()

        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert lines == ["子曰學而時習之不亦說乎"]

    def test_prepare_corpus_warns_when_sampling_disabled(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """句读片段数不多于 input_sentence_size 时发出警告。"""
        _write_corpus(tmp_path, ["子曰。學之。"])
        trainer = _make_trainer(tmp_path, input_sentence_size=100)

        with caplog.at_level(logging.WARNING):
            trainer.prepare_corpus()

        assert any("采样未生效" in record.message for record in caplog.records)

    def test_prepare_corpus_missing_input_raises(self, tmp_path: Path) -> None:
        """语料文件不存在时抛出 FileNotFoundError。"""
        trainer = _make_trainer(tmp_path)

        with pytest.raises(FileNotFoundError):
            trainer.prepare_corpus()
