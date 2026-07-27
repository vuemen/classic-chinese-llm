"""数据去重器测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from classic_chinese_llm.data.deduplicator import (
    Deduplicator,
    DeduplicatorConfig,
    _char_shingles,
    _compute_sha256,
    _exact_dedup,
)


class TestCharShingles:
    """字符级 shingle 生成测试。"""

    def test_basic_5gram(self) -> None:
        """基本 5-gram 生成。"""
        shingles = _char_shingles("子曰学而时习之", k=5)
        assert "子曰学而时" in shingles
        assert "学而时习之" in shingles
        assert len(shingles) == 3  # "子曰学而时", "曰学而时习", "学而时习之"

    def test_newlines_handled(self) -> None:
        """换行符被空格替换。"""
        shingles = _char_shingles("子曰学\n而时习之", k=5)
        assert "\n" not in "".join(shingles)

    def test_short_text_returns_empty(self) -> None:
        """文本短于 k 时返回空集合。"""
        shingles = _char_shingles("短", k=5)
        assert len(shingles) == 0


class TestSHA256:
    """SHA-256 哈希测试。"""

    def test_same_text_same_hash(self) -> None:
        """相同文本产生相同哈希。"""
        h1 = _compute_sha256("子曰学而时习之")
        h2 = _compute_sha256("子曰学而时习之")
        assert h1 == h2

    def test_different_text_different_hash(self) -> None:
        """不同文本产生不同哈希。"""
        h1 = _compute_sha256("子曰学而时习之")
        h2 = _compute_sha256("有朋自远方来")
        assert h1 != h2

    def test_hash_is_hex_string(self) -> None:
        """哈希是 64 字符的十六进制字符串。"""
        h = _compute_sha256("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestExactDedup:
    """精确去重测试。"""

    def test_identical_records_deduped(self) -> None:
        """完全相同的记录被去重。"""
        records = [
            {"text": "子曰学而时习之", "id": "1"},
            {"text": "子曰学而时习之", "id": "2"},
            {"text": "有朋自远方来", "id": "3"},
        ]
        deduped, removed = _exact_dedup(records)
        assert removed == 1
        assert len(deduped) == 2

    def test_all_unique_no_removal(self) -> None:
        """全部不同的记录不被删除。"""
        records = [
            {"text": "文本A", "id": "1"},
            {"text": "文本B", "id": "2"},
            {"text": "文本C", "id": "3"},
        ]
        deduped, removed = _exact_dedup(records)
        assert removed == 0
        assert len(deduped) == 3


class TestDeduplicatorPipeline:
    """Deduplicator 完整管道测试。"""

    def _make_jsonl(self, records: list[dict[str, Any]], path: Path) -> Path:
        file_path = path / "test_input.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return file_path

    def test_exact_dedup_pipeline(self, temp_dir: Path) -> None:
        """端到端精确去重测试。"""
        input_path = self._make_jsonl(
            [
                {"text": "子曰学而时习之不亦说乎", "source": "a"},
                {"text": "子曰学而时习之不亦说乎", "source": "b"},  # 重复
                {"text": "有朋自远方来不亦乐乎", "source": "c"},
            ],
            temp_dir,
        )
        output_path = temp_dir / "deduplicated.jsonl"

        config = DeduplicatorConfig(enable_approx_dedup=False)
        dedup = Deduplicator(config)
        stats = dedup.deduplicate(input_path, output_path)

        assert stats.input_count == 3
        assert stats.exact_removed == 1
        assert stats.after_exact == 2

    def test_empty_input_no_error(self, temp_dir: Path) -> None:
        """空输入不报错。"""
        input_path = self._make_jsonl([], temp_dir)
        output_path = temp_dir / "deduplicated.jsonl"

        dedup = Deduplicator()
        stats = dedup.deduplicate(input_path, output_path)

        assert stats.input_count == 0
        assert stats.after_approx == 0

    def test_stats_math(self, temp_dir: Path) -> None:
        """统计数字自洽。"""
        input_path = self._make_jsonl(
            [
                {"text": "A" * 50, "source": "1"},
                {"text": "A" * 50, "source": "2"},  # 精确重复
                {"text": "B" * 50, "source": "3"},
            ],
            temp_dir,
        )
        output_path = temp_dir / "deduplicated.jsonl"

        config = DeduplicatorConfig(enable_approx_dedup=False)
        dedup = Deduplicator(config)
        stats = dedup.deduplicate(input_path, output_path)

        assert stats.input_count == stats.after_exact + stats.exact_removed

    def test_dedup_keeps_valid_jsonl(self, temp_dir: Path) -> None:
        """去重后的输出是合法的 JSONL。"""
        input_path = self._make_jsonl([{"text": "子曰學而時習之", "source": "test"}], temp_dir)
        output_path = temp_dir / "deduplicated.jsonl"

        config = DeduplicatorConfig(enable_approx_dedup=False)
        dedup = Deduplicator(config)
        dedup.deduplicate(input_path, output_path)

        with open(output_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                assert "text" in record

    def test_approx_dedup_similar_docs_grouped(self, temp_dir: Path) -> None:
        """近似去重：高度相似的文档被归为一组保留一条。"""
        # base 是 similar 的子串，Jaccard 接近 1.0
        common = "子曰學而時習之不亦說乎有朋自遠方來不亦樂乎人不知而不慍不亦君子乎"
        base = common * 3
        similar = base + "學而時習"  # 几乎完全相同，只多了 4 个字
        input_path = self._make_jsonl(
            [
                {"text": base, "source": "a"},
                {"text": similar, "source": "b"},
                {"text": "道可道非常道名可名非常名無名天地之始有物混成先天地生", "source": "c"},
            ],
            temp_dir,
        )
        output_path = temp_dir / "deduplicated.jsonl"

        config = DeduplicatorConfig(num_perm=128, jaccard_threshold=0.85, enable_exact_dedup=True)
        dedup = Deduplicator(config)
        stats = dedup.deduplicate(input_path, output_path)

        # 前两篇高度相似应归为一组，第三篇独立
        assert stats.approx_removed >= 1
        assert stats.after_approx == 2

    def test_approx_dedup_dissimilar_kept_separate(self, temp_dir: Path) -> None:
        """近似去重：不相似的文档各自独立保留。"""
        docs = [
            {"text": "子曰學而時習之不亦說乎有朋自遠方來", "source": "1"},
            {"text": "孟子見梁惠王王曰叟不遠千里而來亦將有以利吾國乎", "source": "2"},
            {"text": "道可道非常道名可名非常名無名天地之始", "source": "3"},
        ]
        input_path = self._make_jsonl(docs, temp_dir)
        output_path = temp_dir / "deduplicated.jsonl"

        config = DeduplicatorConfig(num_perm=128, jaccard_threshold=0.85)
        dedup = Deduplicator(config)
        stats = dedup.deduplicate(input_path, output_path)

        # 三篇完全不同，全部保留
        assert stats.after_approx == 3
