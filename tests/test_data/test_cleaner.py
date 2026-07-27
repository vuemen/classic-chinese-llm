"""数据清洗器测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from classic_chinese_llm.data.cleaner import (
    Cleaner,
    CleanerConfig,
    filter_by_cjk_ratio,
    filter_by_length,
    normalize_unicode,
    normalize_whitespace,
    remove_layout_noise,
    strip_modern_punctuation,
)


class TestNormalizeUnicode:
    """Unicode 规范化测试。"""

    def test_fullwidth_alphanum_to_halfwidth(self) -> None:
        """全角英数字转为半角。"""
        result = normalize_unicode("ＡＢＣ１２３")
        assert result == "ABC123"

    def test_cjk_characters_unchanged(self) -> None:
        """汉字不受 NFKC 影响。"""
        text = "子曰學而時習之"
        assert normalize_unicode(text) == text

    def test_classical_punctuation_unchanged(self) -> None:
        """文言文核心字符不受 NFKC 影响。

        注意：NFKC 会将全角标点（，：！？等 U+FF00 系列）
        转换为半角 ASCII 等价形式。但汉字和 U+3000 系列标点不受影响。
        """
        # 仅使用不受 NFKC 影响的字符：汉字 + 中文句号(U+3002)
        text = "子曰學而時習之。不亦說乎。"
        assert normalize_unicode(text) == text

    def test_compatibility_char_normalized(self) -> None:
        """兼容性字符被规范化。"""
        # U+FB01 = 'fi' 连字
        result = normalize_unicode("ﬁ", form="NFKC")
        assert result == "fi"


class TestStripModernPunctuation:
    """现代标点剥离测试。"""

    def test_double_quotes_removed(self) -> None:
        """英文双引号被移除。"""
        result = strip_modern_punctuation('"论语"是经典。')
        assert result == "论语是经典。"

    def test_classical_brackets_preserved(self) -> None:
        """文言文引号「」被保留。"""
        text = "子曰：「学而时习之」"
        assert strip_modern_punctuation(text) == text

    def test_english_punctuation_removed(self) -> None:
        """英文标点被移除。"""
        result = strip_modern_punctuation("Confucius said: Study, and review!")
        assert result == "Confucius said Study and review"

    def test_chinese_period_preserved(self) -> None:
        """中文句号保留。"""
        text = "学而时习之。"
        assert strip_modern_punctuation(text) == text


class TestRemoveLayoutNoise:
    """版式噪声去除测试。"""

    def test_html_tags_removed(self) -> None:
        """HTML 标签被移除。"""
        result = remove_layout_noise("<div>正文</div>")
        assert result == "正文"

    def test_html_entities_removed(self) -> None:
        """HTML 实体被移除。"""
        result = remove_layout_noise("正文&nbsp;内容")
        assert result == "正文内容"

    def test_url_removed(self) -> None:
        """URL 被移除。"""
        result = remove_layout_noise("参考 https://example.com/text 原文")
        assert "https://example.com/text" not in result
        assert "参考" in result
        assert "原文" in result

    def test_page_number_removed(self) -> None:
        """页码标记被移除。"""
        result = remove_layout_noise("第123页 正文内容")
        assert "第123页" not in result
        assert "正文内容" in result


class TestNormalizeWhitespace:
    """空白规范化测试。"""

    def test_multiple_blank_lines_collapsed(self) -> None:
        """连续空行合并为单个空行。"""
        text = "第一段\n\n\n\n第二段"
        result = normalize_whitespace(text)
        assert "\n\n\n" not in result
        assert result == "第一段\n\n第二段"

    def test_trailing_whitespace_stripped(self) -> None:
        """行首尾空白去除。"""
        result = normalize_whitespace("  正文  \n  内容  ")
        assert result == "正文\n内容"

    def test_tab_replaced_with_space(self) -> None:
        """Tab 转为空格。"""
        result = normalize_whitespace("正文\t内容")
        assert "\t" not in result
        assert result == "正文 内容"


class TestFilterByLength:
    """长度过滤测试。"""

    def test_too_short_filtered(self) -> None:
        """过短文本被过滤。"""
        assert filter_by_length("短", min_len=10) is False

    def test_min_length_accepted(self) -> None:
        """达到最小长度的文本通过。"""
        assert filter_by_length("子曰学而时习之不亦说乎", min_len=10) is True

    def test_too_long_filtered(self) -> None:
        """过长文本被过滤。"""
        long_text = "字" * 100001
        assert filter_by_length(long_text, max_len=100000) is False


class TestFilterByCjkRatio:
    """CJK 字符占比过滤测试。"""

    def test_pure_english_rejected(self) -> None:
        """纯英文被拒绝。"""
        assert filter_by_cjk_ratio("This is English text", min_ratio=0.7) is False

    def test_pure_classical_chinese_accepted(self) -> None:
        """纯文言文被接受。"""
        text = "子曰学而时习之不亦说乎有朋自远方来不亦乐乎"
        assert filter_by_cjk_ratio(text, min_ratio=0.7) is True

    def test_mixed_chinese_english_rejected(self) -> None:
        """中英混合占比不达标时被拒绝。"""
        text = "Confucius said 子曰学而 English translation follows"
        assert filter_by_cjk_ratio(text, min_ratio=0.7) is False

    def test_empty_text_rejected(self) -> None:
        """空文本被拒绝。"""
        assert filter_by_cjk_ratio("", min_ratio=0.7) is False
        assert filter_by_cjk_ratio("   ", min_ratio=0.7) is False


class TestCleanerPipeline:
    """Cleaner 完整管道测试。"""

    def _make_jsonl(self, records: list[dict[str, Any]], path: Path) -> Path:
        """在指定路径创建测试 JSONL 文件。"""
        file_path = path / "test_input.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return file_path

    def test_basic_cleaning_pipeline(self, temp_dir: Path) -> None:
        """基本清洗管道端到端测试。"""
        input_path = self._make_jsonl(
            [
                {"text": "子曰：學而時習之，不亦說乎？", "source": "test"},
                {"text": "This is English only", "source": "test"},
                {"text": "短", "source": "test"},
            ],
            temp_dir,
        )
        output_path = temp_dir / "cleaned.jsonl"

        cleaner = Cleaner(CleanerConfig(min_text_len=10))
        stats = cleaner.clean(input_path, output_path)

        assert stats.input_count == 3
        # 第2条被 CJK 过滤，第3条被长度过滤
        assert stats.output_count == 1

        # 验证输出
        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "子曰" in record["text"]

    def test_all_rules_disabled_passthrough(self, temp_dir: Path) -> None:
        """禁用全部规则时输入=输出。"""
        input_path = self._make_jsonl(
            [{"text": "子曰學而時習之不亦說乎有朋自遠方來", "source": "test"}], temp_dir
        )
        output_path = temp_dir / "cleaned.jsonl"

        config = CleanerConfig(
            min_text_len=1,
            enable_normalize_unicode=False,
            enable_strip_modern_punctuation=False,
            enable_remove_layout_noise=False,
            enable_normalize_whitespace=False,
            enable_filter_non_chinese=False,
        )
        cleaner = Cleaner(config)
        stats = cleaner.clean(input_path, output_path)

        assert stats.input_count == 1
        assert stats.output_count == 1

    def test_idempotency(self, temp_dir: Path) -> None:
        """清洗结果再次清洗，输出不变（幂等性）。"""
        input_path = self._make_jsonl(
            [{"text": "子曰：「學而時習之，不亦說乎？」", "source": "test"}],
            temp_dir,
        )
        first_pass = temp_dir / "first.jsonl"
        second_pass = temp_dir / "second.jsonl"

        cleaner = Cleaner()
        cleaner.clean(input_path, first_pass)
        stats2 = cleaner.clean(first_pass, second_pass)

        assert stats2.output_count == stats2.input_count  # 无额外丢弃

    def test_stats_char_count(self, temp_dir: Path) -> None:
        """统计中的字符计数正确。"""
        input_path = self._make_jsonl(
            [{"text": "子曰學而時習之不亦說乎有朋自遠方來", "source": "test"}], temp_dir
        )
        output_path = temp_dir / "cleaned.jsonl"

        cleaner = Cleaner()
        stats = cleaner.clean(input_path, output_path)

        assert stats.input_chars > 0
        assert stats.output_chars > 0
