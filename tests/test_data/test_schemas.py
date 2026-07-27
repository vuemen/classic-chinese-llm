"""SourceDocument 数据模型测试。"""

from __future__ import annotations

import json

from classic_chinese_llm.data.schemas import SourceDocument


class TestSourceDocument:
    """SourceDocument 数据类测试。"""

    def test_default_values(self) -> None:
        """默认字段值正确。"""
        doc = SourceDocument(text="子曰学而时习之", source="test")
        assert doc.text == "子曰学而时习之"
        assert doc.source == "test"
        assert doc.title == ""
        assert doc.author == ""
        assert doc.era == ""
        assert doc.genre == ""
        assert doc.url == ""
        assert doc.chapter == ""
        assert doc.metadata == {}
        assert doc.collected_at != ""

    def test_to_jsonl_line_valid_json(self) -> None:
        """to_jsonl_line 返回合法的 JSON 字符串。"""
        doc = SourceDocument(
            text="论语·学而篇",
            source="test",
            title="学而",
            author="孔子",
            era="先秦",
            genre="经",
        )
        line = doc.to_jsonl_line()
        parsed = json.loads(line)
        assert parsed["text"] == "论语·学而篇"
        assert parsed["source"] == "test"
        assert parsed["title"] == "学而"
        assert parsed["era"] == "先秦"
        assert parsed["genre"] == "经"

    def test_to_jsonl_line_ensure_ascii_false(self) -> None:
        """to_jsonl_line 保留中文字符（不转义为 \\uXXXX）。"""
        doc = SourceDocument(text="论语", source="test")
        line = doc.to_jsonl_line()
        assert "论语" in line
        assert "\\u8bba" not in line

    def test_full_fields_roundtrip(self) -> None:
        """完整字段通过 JSONL 往返不丢失。"""
        original = SourceDocument(
            text="子曰：学而时习之，不亦说乎？",
            source="lunyu",
            title="学而篇第一",
            author="孔子",
            era="先秦",
            genre="经",
            url="https://example.com/lunyu",
            chapter="卷一",
            metadata={"version": "1.0"},
        )
        line = original.to_jsonl_line()
        reloaded = json.loads(line)
        assert reloaded["text"] == original.text
        assert reloaded["title"] == original.title
        assert reloaded["metadata"]["version"] == "1.0"

    def test_empty_metadata_is_empty_dict(self) -> None:
        """元信息默认为空字典。"""
        doc = SourceDocument(text="test", source="test")
        assert doc.metadata == {}
