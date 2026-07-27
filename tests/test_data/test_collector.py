"""数据采集器测试。"""

from __future__ import annotations

import json
from pathlib import Path

from classic_chinese_llm.data.collector import Collector
from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.data.sources.base import BaseSource


class _MockSource(BaseSource):
    """测试用模拟数据源。"""

    name = "mock"
    display_name = "模拟来源"

    def __init__(self, data_dir: Path, docs: list[SourceDocument] | None = None) -> None:
        super().__init__(data_dir)
        self._docs = docs or []

    def discover(self, raw_dir: Path) -> list[Path]:
        """发现模拟文件。"""
        src_dir = raw_dir / self.name
        src_dir.mkdir(parents=True, exist_ok=True)
        # 创建测试文件
        test_file = src_dir / "test.txt"
        test_file.write_text("子曰学而时习之\n\n有朋自远方来\n\n人不知而不愠", encoding="utf-8")
        return [test_file]

    def parse(self, file_path: Path) -> list[SourceDocument]:
        """解析为固定的测试文档。"""
        if self._docs:
            return self._docs
        return [
            SourceDocument(text="子曰学而时习之", source=self.name, title="学而"),
            SourceDocument(text="有朋自远方来", source=self.name, title="学而"),
            SourceDocument(text="人不知而不愠", source=self.name, title="学而"),
        ]

    def validate(self, doc: SourceDocument) -> bool:
        """正文长度 >= 5 字符。"""
        return len(doc.text.strip()) >= 5


class _FailingSource(BaseSource):
    """模拟解析失败的数据源。"""

    name = "failing"
    display_name = "失败来源"

    def discover(self, raw_dir: Path) -> list[Path]:
        return []

    def parse(self, file_path: Path) -> list[SourceDocument]:
        raise RuntimeError("模拟错误")


class TestCollector:
    """Collector 测试。"""

    def test_basic_collection(self, temp_dir: Path) -> None:
        """基本采集流程测试。"""
        raw_dir = temp_dir / "raw"
        output_dir = temp_dir / "processed"

        mock = _MockSource(raw_dir)
        collector = Collector([mock])
        output_path = collector.run(raw_dir=raw_dir, output_dir=output_dir)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert record["source"] == "mock"
            assert len(record["text"]) >= 5

    def test_empty_source_skipped(self, temp_dir: Path) -> None:
        """无文件的数据源被跳过不报错。"""
        raw_dir = temp_dir / "raw"
        output_dir = temp_dir / "processed"

        failing = _FailingSource(raw_dir)
        collector = Collector([failing])
        output_path = collector.run(raw_dir=raw_dir, output_dir=output_dir)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            assert len(f.readlines()) == 0

    def test_multiple_sources(self, temp_dir: Path) -> None:
        """多个数据源同时采集。"""
        raw_dir = temp_dir / "raw"
        output_dir = temp_dir / "processed"

        mock1 = _MockSource(raw_dir)
        mock2 = _MockSource(raw_dir)
        mock2.name = "mock2"
        mock2.display_name = "模拟来源2"
        collector = Collector([mock1, mock2])
        output_path = collector.run(raw_dir=raw_dir, output_dir=output_dir)

        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()
        # 两个来源各产出了记录
        assert len(lines) > 0

    def test_output_is_valid_jsonl(self, temp_dir: Path) -> None:
        """输出文件的每一行都是合法的 JSON。"""
        raw_dir = temp_dir / "raw"
        output_dir = temp_dir / "processed"

        mock = _MockSource(raw_dir)
        collector = Collector([mock])
        output_path = collector.run(raw_dir=raw_dir, output_dir=output_dir)

        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)  # 不抛异常即为通过

    def test_documents_preserve_metadata(self, temp_dir: Path) -> None:
        """文档元信息在采集过程中保留。"""
        raw_dir = temp_dir / "raw"
        output_dir = temp_dir / "processed"

        custom_docs = [
            SourceDocument(
                text="子程子曰：大學，孔氏之遺書。",
                source="mock",
                title="大學章句序",
                author="朱熹",
                era="宋",
                genre="经",
            ),
        ]
        mock = _MockSource(raw_dir, docs=custom_docs)
        collector = Collector([mock])
        output_path = collector.run(raw_dir=raw_dir, output_dir=output_dir)

        with open(output_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["title"] == "大學章句序"
        assert record["author"] == "朱熹"
        assert record["era"] == "宋"
        assert record["genre"] == "经"
