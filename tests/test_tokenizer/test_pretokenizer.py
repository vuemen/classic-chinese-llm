"""文言文预分词器测试。"""

from __future__ import annotations

from classic_chinese_llm.tokenizer.pretokenizer import ClassicalChinesePreTokenizer


class TestClassicalChinesePreTokenizer:
    """ClassicalChinesePreTokenizer 单元测试。"""

    # ─── 强分隔符测试 ──────────────────────────────────────────────

    def test_split_on_period(self) -> None:
        """句号处断句。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("子曰學而時習之。不亦說乎。")
        texts = [part for part, _ in result]
        assert len(texts) >= 2
        assert "不亦說乎。" in texts

    def test_split_on_question_mark(self) -> None:
        """问号处断句。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("不亦說乎？有朋自遠方來。")
        texts = [part for part, _ in result]
        assert len(texts) >= 2
        assert any("不亦說乎？" in t for t in texts)

    def test_split_on_exclamation(self) -> None:
        """感叹号处断句。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("善哉！吾聞之矣。")
        texts = [part for part, _ in result]
        assert len(texts) >= 2
        assert "善哉！" in texts[0]

    def test_split_on_semicolon(self) -> None:
        """分号处断句。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("學而不思則罔；思而不學則殆。")
        texts = [part for part, _ in result]
        assert len(texts) >= 2

    # ─── 弱分隔符测试 ──────────────────────────────────────────────

    def test_split_on_comma(self) -> None:
        """逗号处断句。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("子曰，學而時習之，不亦說乎。")
        texts = [part for part, _ in result]
        assert len(texts) >= 2

    def test_split_on_enumeration_comma(self) -> None:
        """顿号处断句。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("詩、書、禮、樂皆通。")
        texts = [part for part, _ in result]
        assert len(texts) >= 2

    def test_split_on_colon(self) -> None:
        """冒号处断句。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("子曰：學而時習之。")
        texts = [part for part, _ in result]
        assert len(texts) >= 2

    # ─── 标点保留测试 ──────────────────────────────────────────────

    def test_punctuation_preserved(self) -> None:
        """所有标点保留在结果中。"""
        pretok = ClassicalChinesePreTokenizer()
        text = "子曰：「學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？」"
        result = pretok(text)
        reconstructed = "".join(part for part, _ in result)
        assert reconstructed == text

    def test_classical_brackets_preserved(self) -> None:
        """文言文引号「」不被分割。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("子曰：「學而時習之。」")
        texts = [part for part, _ in result]
        # 「」不应作为分隔符
        assert any("「" in t for t in texts)
        assert any("」" in t for t in texts)

    # ─── 边界情况 ──────────────────────────────────────────────────

    def test_empty_input(self) -> None:
        """空输入返回空列表。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("")
        assert result == []

    def test_no_punctuation(self) -> None:
        """无标点文本返回单元素列表。"""
        pretok = ClassicalChinesePreTokenizer()
        text = "子曰學而時習之不亦說乎有朋自遠方來"
        result = pretok(text)
        assert len(result) == 1
        assert result[0][0] == text

    def test_pure_punctuation(self) -> None:
        """仅含标点的文本正确处理。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("。！？")
        # 每个标点后都分割，全部非空片段都应保留
        assert len(result) >= 1

    def test_whitespace_only(self) -> None:
        """仅含空白的文本。"""
        pretok = ClassicalChinesePreTokenizer()
        result = pretok("   \n  ")
        assert len(result) == 1
        assert result[0][0] == "   \n  "

    def test_byte_offset_correct(self) -> None:
        """字节偏移计算正确。"""
        pretok = ClassicalChinesePreTokenizer()
        text = "子曰。學之。"
        result = pretok(text)
        # 第一个片段 "子曰。" 偏移应为 0
        assert result[0][1] == 0
        # 所有偏移之和应等于文本的 UTF-8 字节长度
        total_bytes = sum(len(part.encode("utf-8")) for part, _ in result)
        assert total_bytes == len(text.encode("utf-8"))

    def test_pre_tokenize_convenience(self) -> None:
        """pre_tokenize 便捷方法返回纯文本列表。"""
        pretok = ClassicalChinesePreTokenizer()
        texts = pretok.pre_tokenize("子曰。學之。")
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)
        assert len(texts) >= 2

    def test_mixed_strong_weak_punctuation(self) -> None:
        """强弱标点混合文本正确分割。"""
        pretok = ClassicalChinesePreTokenizer()
        text = "子曰：學而時習之，不亦說乎？有朋自遠方來，不亦樂乎。"
        result = pretok(text)
        texts = [part for part, _ in result]
        # 在 ：，？， 四处标点后都应分割
        assert len(texts) >= 4

    def test_long_classical_text(self) -> None:
        """长篇文言文分割正确。"""
        pretok = ClassicalChinesePreTokenizer()
        text = (
            "大學之道，在明明德，在親民，在止於至善。"
            "知止而後有定；定而後能靜；靜而後能安；"
            "安而後能慮；慮而後能得。"
        )
        result = pretok(text)
        texts = [part for part, _ in result]
        assert len(texts) >= 5
        # 验证可以无损重建
        assert "".join(texts) == text
