"""评测指标计算函数。

每个函数接受 predictions (list[str]) 和 references (list[list[str]])，
返回 float 类型的指标值 (0.0 ~ 1.0，perplexity 除外)。
"""

from __future__ import annotations

import math
import re
from collections import Counter


def calc_perplexity(loss: float) -> float:
    """将 cross-entropy loss 转换为 perplexity。

    PPL = exp(loss)

    Args:
        loss: 平均 cross-entropy loss 值。

    Returns:
        float: perplexity 值 (≥ 1.0)。

    Raises:
        ValueError: 如果 loss 为负数。
    """
    if loss < 0:
        raise ValueError("loss 必须为非负数")
    return math.exp(loss)


def calc_bleu(
    predictions: list[str],
    references: list[list[str]],
    max_n: int = 4,
) -> float:
    """计算 corpus-level BLEU-n 分数。

    使用 smoothing=1 避免短文本零分问题。

    Args:
        predictions: 预测文本列表。
        references: 参考文本列表 (每条预测可对应多个参考)。
        max_n: n-gram 最大阶数 (默认 4)。

    Returns:
        float: BLEU 分数 (0.0 ~ 1.0)。
    """
    if not predictions or all(len(p) == 0 for p in predictions):
        return 0.0

    precisions: list[float] = []
    total_pred_chars = 0

    for n in range(1, max_n + 1):
        matched_count = 0
        total_count = 0

        for pred, refs in zip(predictions, references, strict=False):
            pred_ngrams = _char_ngrams(pred, n)
            total_count += len(pred_ngrams)
            if not pred_ngrams:
                continue

            # 取所有参考中最佳匹配的计数
            best_match = 0
            for ref in refs:
                ref_ngrams = _char_ngrams(ref, n)
                ref_counter = Counter(ref_ngrams)
                match = sum(
                    min(pred_ngrams.get(ng, 0), ref_counter.get(ng, 0)) for ng in pred_ngrams
                )
                best_match = max(best_match, match)
            matched_count += best_match

        if total_count == 0:
            precisions.append(0.0)
        else:
            # smoothing=1: 分子分母各加 1
            precisions.append((matched_count + 1) / (total_count + 1))

        total_pred_chars += sum(len(p) for p in predictions)

    if any(p == 0.0 for p in precisions):
        return 0.0

    # 几何平均
    bleu_score = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    # 长度惩罚 (Brevity Penalty)
    total_ref_chars = sum(min(len(r) for r in refs) if refs else 0 for refs in references)
    if total_pred_chars < total_ref_chars and total_pred_chars > 0:
        bp = math.exp(1 - total_ref_chars / total_pred_chars)
    else:
        bp = 1.0

    return min(bp * bleu_score, 1.0)


def calc_rouge_l(
    predictions: list[str],
    references: list[list[str]],
) -> float:
    """计算 ROUGE-L (Longest Common Subsequence) F1 分数。

    ROUGE-L 使用最长公共子序列 (LCS) 来衡量生成文本
    与参考文本之间的相似度。

    Args:
        predictions: 预测文本列表。
        references: 参考文本列表。

    Returns:
        float: ROUGE-L F1 分数 (0.0 ~ 1.0)。
    """
    if not predictions or all(len(p) == 0 for p in predictions):
        return 0.0

    total_f1 = 0.0
    sample_count = 0

    for pred, refs in zip(predictions, references, strict=False):
        if len(pred) == 0:
            total_f1 += 0.0
            sample_count += 1
            continue

        best_f1 = 0.0
        for ref in refs:
            lcs_len = _lcs_length(pred, ref)
            if lcs_len == 0:
                continue
            recall = lcs_len / len(ref)
            precision = lcs_len / len(pred)
            if recall + precision == 0:
                continue
            f1 = 2 * recall * precision / (recall + precision)
            best_f1 = max(best_f1, f1)

        total_f1 += best_f1
        sample_count += 1

    if sample_count == 0:
        return 0.0
    return total_f1 / sample_count


def calc_char_accuracy(
    predictions: list[str],
    references: list[list[str]],
) -> float:
    """逐字符匹配准确率。

    每个预测与最佳匹配参考计算字符级完全匹配率，
    取所有样本的平均值。

    Args:
        predictions: 预测文本列表。
        references: 参考文本列表。

    Returns:
        float: 字符准确率 (0.0 ~ 1.0)。
    """
    if not predictions or all(len(p) == 0 for p in predictions):
        return 0.0

    total_acc = 0.0
    sample_count = 0

    for pred, refs in zip(predictions, references, strict=False):
        if len(pred) == 0:
            total_acc += 0.0
            sample_count += 1
            continue

        best_acc = 0.0
        for ref in refs:
            # 逐字符比较，以较短的为分母（惩罚漏字）
            match_count = sum(1 for a, b in zip(pred, ref, strict=False) if a == b)
            denominator = max(len(pred), len(ref))
            if denominator > 0:
                acc = match_count / denominator
                best_acc = max(best_acc, acc)

        total_acc += best_acc
        sample_count += 1

    if sample_count == 0:
        return 0.0
    return total_acc / sample_count


# ─── 文言文专用评分 ────────────────────────────────────────────────────────


# 常见文言虚词
_FUNCTION_WORDS = set(
    "之乎者也矣焉哉耳邪耶歟與於為以而則乃其夫蓋且然雖若故所以是以可以"
    "乎哉而已云爾者所諸兮噫嘻夫惟"
)


def calc_classical_chinese_score(prediction: str) -> dict[str, float]:
    """文言文质量专用评分。

    检查三个方面:
    1. 虚词密度: 虚词使用频率（适中为佳）
    2. 平均句长: 文言文宜简短（理想 4-8 字/句）
    3. 总分: 综合加权

    Args:
        prediction: 待评分的文言文文本。

    Returns:
        dict: {"虚词密度": float, "平均句长": float, "总分": float}
              每个值在 0.0 ~ 1.0 之间。
    """
    if len(prediction) == 0:
        return {"虚词密度": 0.0, "平均句长": 0.0, "总分": 0.0}

    # 虚词密度
    func_word_count = sum(1 for ch in prediction if ch in _FUNCTION_WORDS)
    func_density = func_word_count / len(prediction)
    # 理想虚词密度约 8%-20%
    func_score = _range_score(func_density, ideal_min=0.08, ideal_max=0.20)

    # 平均句长（按 。！？；断句）
    sentences = re.split(r"[。！？；，\n]", prediction)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        avg_len = sum(len(s) for s in sentences) / len(sentences)
        # 理想句长 4-8 字
        len_score = _range_score(avg_len, ideal_min=4.0, ideal_max=8.0)
    else:
        len_score = 0.0

    # 总分加权
    total_score = 0.5 * func_score + 0.5 * len_score

    return {
        "虚词密度": round(func_score, 4),
        "平均句长": round(len_score, 4),
        "总分": round(total_score, 4),
    }


# ─── 辅助函数 ───────────────────────────────────────────────────────────────


def _char_ngrams(text: str, n: int) -> Counter[str]:
    """将文本拆分为字符级 n-gram。

    Args:
        text: 输入文本。
        n: n-gram 阶数。

    Returns:
        Counter: n-gram 频次计数。
    """
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _lcs_length(a: str, b: str) -> int:
    """计算两个字符串的最长公共子序列长度（动态规划）。

    Args:
        a: 字符串 A。
        b: 字符串 B。

    Returns:
        int: LCS 长度。
    """
    if not a or not b:
        return 0

    # 使用 O(min(|a|,|b|)) 空间的一维 DP
    if len(a) < len(b):
        a, b = b, a

    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[len(b)]


def _range_score(
    value: float,
    ideal_min: float,
    ideal_max: float,
) -> float:
    """根据值是否落在理想区间内计算评分。

    Args:
        value: 实际值。
        ideal_min: 理想区间下限。
        ideal_max: 理想区间上限。

    Returns:
        float: 评分 (0.0 ~ 1.0)，落在区间内得 1.0，
               偏离越远分越低。
    """
    if ideal_min <= value <= ideal_max:
        return 1.0
    if value < ideal_min:
        return max(0.0, value / ideal_min)
    # value > ideal_max
    if ideal_max > 0:
        return max(0.0, ideal_max / value)
    return 0.0
