"""evaluation.metrics 模块的单元测试。"""

from __future__ import annotations

import math

import pytest

from classic_chinese_llm.evaluation.metrics import (
    calc_bleu,
    calc_char_accuracy,
    calc_classical_chinese_score,
    calc_perplexity,
    calc_rouge_l,
)


class TestPerplexity:
    """calc_perplexity 测试。"""

    def test_basic(self) -> None:
        """loss=1.0 → ppl ≈ 2.718。"""
        result = calc_perplexity(1.0)
        assert math.isclose(result, math.e, rel_tol=1e-6)

    def test_zero_loss(self) -> None:
        """loss=0.0 → ppl = 1.0。"""
        result = calc_perplexity(0.0)
        assert result == 1.0

    def test_high_loss(self) -> None:
        """loss=5.0 → ppl ≈ 148.4。"""
        result = calc_perplexity(5.0)
        assert math.isclose(result, math.exp(5.0), rel_tol=1e-6)

    def test_negative_loss_raises(self) -> None:
        """负数 loss 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="loss 必须为非负数"):
            calc_perplexity(-1.0)


class TestBLEU:
    """calc_bleu 测试。"""

    def test_perfect_match(self) -> None:
        """完全匹配应返回接近 1.0。"""
        preds = ["子曰学而时习之不亦说乎"]
        refs = [["子曰学而时习之不亦说乎"]]
        result = calc_bleu(preds, refs)
        assert result > 0.9

    def test_no_match(self) -> None:
        """完全不匹配应返回较低分数。"""
        preds = ["吾日三省吾身"]
        refs = [["子曰学而时习之"]]
        result = calc_bleu(preds, refs)
        # 由于 smoothing=1，即使无共同 n-gram 也有一个很小的 baseline
        assert result < 0.3

    def test_empty_prediction(self) -> None:
        """空预测应返回 0.0。"""
        preds = [""]
        refs = [["参考文本"]]
        result = calc_bleu(preds, refs)
        assert result == 0.0

    def test_multiple_samples(self) -> None:
        """多样本 BLEU 计算。"""
        preds = ["学而时习之", "吾日三省吾身"]
        refs = [["学而时习之不亦说乎"], ["吾日三省吾身"]]
        result = calc_bleu(preds, refs)
        assert 0.0 < result <= 1.0

    def test_multiple_references(self) -> None:
        """多参考 BLEU 应取最佳匹配。"""
        preds = ["学而时习之不亦说乎"]
        refs = [["学而时习之不亦说乎", "学而时习之"]]
        result = calc_bleu(preds, refs)
        assert result > 0.9


class TestROUGEL:
    """calc_rouge_l 测试。"""

    def test_perfect_match(self) -> None:
        """完全匹配应返回接近 1.0。"""
        preds = ["学而时习之不亦说乎"]
        refs = [["学而时习之不亦说乎"]]
        result = calc_rouge_l(preds, refs)
        assert math.isclose(result, 1.0, rel_tol=1e-6)

    def test_no_match(self) -> None:
        """无共同字符应返回 0.0。"""
        preds = ["天地玄黄"]
        refs = [["宇宙洪荒"]]
        result = calc_rouge_l(preds, refs)
        assert result == 0.0

    def test_partial_match(self) -> None:
        """部分匹配应在 0 和 1 之间。"""
        preds = ["学而时习之"]
        refs = [["学而时习之不亦说乎"]]
        result = calc_rouge_l(preds, refs)
        assert 0.0 < result < 1.0

    def test_empty_prediction(self) -> None:
        """空预测应返回 0.0。"""
        preds = [""]
        refs = [["参考"]]
        result = calc_rouge_l(preds, refs)
        assert result == 0.0

    def test_multiple_references(self) -> None:
        """多参考取最佳 ROUGE-L F1。"""
        preds = ["学而时习之"]
        refs = [["学而时习之不亦说乎", "学而时习之"]]
        result = calc_rouge_l(preds, refs)
        assert result > 0.9


class TestCharAccuracy:
    """calc_char_accuracy 测试。"""

    def test_perfect_match(self) -> None:
        """完全匹配返回 1.0。"""
        preds = ["天地玄黄宇宙洪荒"]
        refs = [["天地玄黄宇宙洪荒"]]
        result = calc_char_accuracy(preds, refs)
        assert result == 1.0

    def test_no_match(self) -> None:
        """无匹配返回 0.0。"""
        preds = ["天地"]
        refs = [["春秋"]]
        result = calc_char_accuracy(preds, refs)
        assert result == 0.0

    def test_partial_match(self) -> None:
        """部分匹配。"""
        preds = ["天地玄黄"]
        refs = [["天地宇宙"]]
        result = calc_char_accuracy(preds, refs)
        assert 0.0 < result < 1.0

    def test_different_length(self) -> None:
        """不同长度时以较短的为分母（惩罚漏字）。"""
        preds = ["天地"]
        refs = [["天地玄黄"]]
        result = calc_char_accuracy(preds, refs)
        assert 0.0 < result < 1.0

    def test_multiple_references_best_match(self) -> None:
        """多参考取最佳。"""
        preds = ["天地玄黄"]
        refs = [["天地宇宙", "天地玄黄"]]
        result = calc_char_accuracy(preds, refs)
        assert result == 1.0

    def test_empty_prediction(self) -> None:
        """空预测返回 0.0。"""
        preds = [""]
        refs = [["参考"]]
        result = calc_char_accuracy(preds, refs)
        assert result == 0.0


class TestClassicalChineseScore:
    """calc_classical_chinese_score 测试。"""

    def test_returns_expected_keys(self) -> None:
        """应返回预定义指标字段。"""
        text = "子曰：学而时习之，不亦说乎？"
        scores = calc_classical_chinese_score(text)
        assert "虚词密度" in scores
        assert "平均句长" in scores
        assert "典故覆盖率" in scores
        assert "总分" in scores
        # 所有值在 [0, 1] 范围内
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_empty_text(self) -> None:
        """空文本各项指标为 0。"""
        scores = calc_classical_chinese_score("")
        assert all(v == 0.0 for v in scores.values())
        assert "典故覆盖率" in scores

    def test_high_quality_text(self) -> None:
        """含虚词和典故的文言文文本得分应较高。"""
        text = "子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？"
        scores = calc_classical_chinese_score(text)
        assert scores["总分"] > 0.1

    def test_with_allusion(self) -> None:
        """包含典故关键词的文本典故覆盖率 > 0。"""
        text = "孔子曰：君子喻於義，小人喻於利。此乃仁義之道也。"
        scores = calc_classical_chinese_score(text)
        assert scores["典故覆盖率"] > 0.0

    def test_mixed_text(self) -> None:
        """混合现代白话文本得分较低。"""
        classical = "子曰：学而时习之，不亦说乎？"
        modern = "今天天气真好我们去公园玩吧"
        classical_scores = calc_classical_chinese_score(classical)
        modern_scores = calc_classical_chinese_score(modern)
        assert classical_scores["总分"] >= modern_scores["总分"]
