"""文言文专用预分词器。

按文言文句读标点进行预分词：
- 强分隔符（。！？；）—— 必定断句
- 弱分隔符（，、：）—— 可选断句位置

所有标点保留在分词结果中，不丢弃。

提供两种使用方式：
1. ClassicalChinesePreTokenizer().__call__(text) — 纯 Python，返回 (text, offset) 列表
2. create_pretokenizer() — 适配为 HF tokenizers PreTokenizer 接口
"""

from __future__ import annotations

import re
from typing import ClassVar

from tokenizers import pre_tokenizers


class ClassicalChinesePreTokenizer:
    """按文言文句读标点进行预分词。

    分两层:
    1. 强分隔符（。！？；）—— 必定在此处断句
    2. 弱分隔符（，、：）—— 可选断句位置，由 tokenizer 最终决定

    使用方式:
        pretok = ClassicalChinesePreTokenizer()
        splits = pretok("子曰：「學而時習之，不亦說乎？」")
        # → [("子曰：「學而時習之，", 0), ("不亦說乎？", ...), ...]
    """

    STRONG_PUNCT: ClassVar[str] = "。！？；"
    """强分隔符：必定断句。"""

    WEAK_PUNCT: ClassVar[str] = "，、："
    """弱分隔符：可选断句位置。"""

    EXTRA_PUNCT: ClassVar[str] = "…—"
    """额外处理的标点符号。"""

    def __init__(self) -> None:
        all_punct = self.STRONG_PUNCT + self.WEAK_PUNCT + self.EXTRA_PUNCT
        # 正向后顾断言：在标点之后分割，标点属于前一段
        self._pattern = re.compile(rf"(?<=[{re.escape(all_punct)}])")

    def __call__(self, text: str) -> list[tuple[str, int]]:
        """在标点处分割文本，返回 (片段, 位移) 列表。

        每个 tuple 为 (片段文本, 在原始文本中的字节偏移)。
        位移信息供 HF tokenizers 的对齐追踪使用。

        Args:
            text: 输入文言文文本。

        Returns:
            (片段文本, 字节偏移) 列表。若输入为空或无标点，
            返回包含整个文本的单元素列表。
        """
        if not text:
            return []

        parts = self._pattern.split(text)

        # 过滤空字符串，计算字节偏移
        results: list[tuple[str, int]] = []
        byte_offset = 0
        for part in parts:
            if part:
                results.append((part, byte_offset))
                byte_offset += len(part.encode("utf-8"))

        # 如果没有找到任何分割点，返回整个文本
        if not results:
            results.append((text, 0))

        return results

    def pre_tokenize(self, text: str) -> list[str]:
        """便捷方法：仅返回文本片段列表，不含字节偏移。

        Args:
            text: 输入文本。

        Returns:
            文本片段列表。
        """
        return [part for part, _ in self(text)]


def create_pretokenizer() -> pre_tokenizers.PreTokenizer:
    """创建 HF tokenizers 兼容的预分词器。

    将 ClassicalChinesePreTokenizer 适配为 tokenizers.PreTokenizer 接口。
    可用于 tokenizer.pre_tokenizer 赋值，或作为独立的预处理步骤。

    注意:
        通过 PreTokenizer.custom() 创建的自定义预分词器不支持 pickle 序列化。
        因此 build_tokenizer() 不使用此适配器（改用 SentencePiece 内置的规范化器）。
        如需文言文标点断句，建议在 tokenizer 编码前手动调用
        ClassicalChinesePreTokenizer().pre_tokenize(text) 进行预处理。

    Returns:
        HF tokenizers PreTokenizer 实例。
    """
    custom = ClassicalChinesePreTokenizer()
    return pre_tokenizers.PreTokenizer.custom(custom)
