"""数据清洗器。

管道式文本清洗：每个规则是独立的 Callable[[str], str]，
按序执行转换后，通过过滤规则筛选。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# ─── 类型别名 ──────────────────────────────────────────────────────────

CleaningRule = Callable[[str], str]
"""清洗规则签名: 接受文本，返回清洗后文本。"""

FilterRule = Callable[[str], bool]
"""过滤规则签名: 接受文本，True=保留，False=丢弃。"""


# ─── 配置 ─────────────────────────────────────────────────────────────


@dataclass
class CleanerConfig:
    """清洗器可配置参数。"""

    min_text_len: int = 10
    max_text_len: int = 100000
    min_cjk_ratio: float = 0.7
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC"

    enable_normalize_unicode: bool = True
    enable_strip_modern_punctuation: bool = True
    enable_remove_layout_noise: bool = True
    enable_normalize_whitespace: bool = True
    enable_filter_non_chinese: bool = True


# ─── 文言文标点白名单 ─────────────────────────────────────────────────

_CLASSICAL_PUNCTUATION = frozenset("。，、；：？！「」『』．·《》〈〉——……")

_PUNCTUATION_CATEGORIES = frozenset({"Po", "Ps", "Pe", "Pi", "Pf", "Pc", "Pd"})


def _is_classical_punct(char: str) -> bool:
    """判断字符是否为文言文合法标点。"""
    return char in _CLASSICAL_PUNCTUATION


def _is_modern_punct(char: str) -> bool:
    """判断字符是否为现代标点（Unicode 标点类别但不在白名单中）。"""
    cat = unicodedata.category(char)
    if cat not in _PUNCTUATION_CATEGORIES:
        return False
    return not _is_classical_punct(char)


# ─── 版式噪声正则 ─────────────────────────────────────────────────────

_RE_PAGE_NUMBER = re.compile(r"(第\s*\d+\s*[页頁])|([pP]\.?\s*\d+)|([—\-]\s*\d+\s*[—\-])")
_RE_ANNOTATION_MARKER = re.compile(r"[①-⑳㈠-㈩㊀-㊉]")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")


# ─── 内置清洗规则 ─────────────────────────────────────────────────────


def normalize_unicode(text: str, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC") -> str:
    """Unicode 规范化: 全角→半角数字/字母，兼容性字符→标准形式。

    NFKC 将全角英数字转为半角（'Ａ'→'A', '１'→'1'），
    文言文核心字符（汉字、古典标点）不受 NFKC 影响。
    """
    return unicodedata.normalize(form, text)


def strip_modern_punctuation(text: str) -> str:
    """移除现代标点。

    删除属于 Unicode 标点类别但不在文言文白名单中的字符。
    示例:
        '"论语"是儒家经典。' → '论语是儒家经典。'
        '子曰：「学而时习之……」' → 保留（文言标点不受影响）
    """
    return "".join(ch for ch in text if not _is_modern_punct(ch))


def remove_layout_noise(text: str) -> str:
    """去除版式噪声：页码、注释标记、HTML 标签、URL。"""
    text = _RE_URL.sub("", text)
    text = _RE_HTML_TAG.sub("", text)
    text = _RE_HTML_ENTITY.sub("", text)
    text = _RE_PAGE_NUMBER.sub("", text)
    text = _RE_ANNOTATION_MARKER.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    """规范化空白字符。

    文言文规则:
    - 行首行尾空白去除
    - 连续空行合并为单个空行
    - Tab 转空格
    """
    text = text.replace("\t", " ")
    lines = [line.strip() for line in text.splitlines()]
    result: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    return "\n".join(result)


# ─── 内置过滤规则 ─────────────────────────────────────────────────────


def filter_by_length(text: str, min_len: int = 10, max_len: int = 100000) -> bool:
    """长度过滤：过短无信息量，过长可能是未分段的大文件。"""
    clean = text.strip()
    return min_len <= len(clean) <= max_len


def filter_by_cjk_ratio(text: str, min_ratio: float = 0.7) -> bool:
    """CJK 汉字占比过滤。

    统计 Unicode CJK 统一汉字区（U+4E00–U+9FFF）字符在
    非空白字符中的占比。低于阈值的文本（英文、日文假名等）被丢弃。
    """
    stripped = text.strip()
    if not stripped:
        return False
    non_space = [ch for ch in stripped if not ch.isspace()]
    if not non_space:
        return False
    cjk_count = sum(1 for ch in non_space if "一" <= ch <= "鿿")
    return (cjk_count / len(non_space)) >= min_ratio


# ─── 清洗编排器 ───────────────────────────────────────────────────────


@dataclass
class CleaningStats:
    """单次清洗的统计信息。"""

    input_count: int = 0
    output_count: int = 0
    filtered_by_length: int = 0
    filtered_by_cjk: int = 0
    input_chars: int = 0
    output_chars: int = 0


class Cleaner:
    """数据清洗编排器。

    持有转换规则列表和过滤规则列表，对 JSONL 输入逐行清洗。

    用法:
        cleaner = Cleaner(CleanerConfig(min_text_len=20))
        stats = cleaner.clean(input_path, output_path)
    """

    def __init__(self, config: CleanerConfig | None = None) -> None:
        self.config = config or CleanerConfig()
        self._transform_rules: list[CleaningRule] = []
        self._filter_rules: list[FilterRule] = []
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """根据配置组装转换管道和过滤管道。"""
        cfg = self.config

        # 转换规则（按顺序执行）
        if cfg.enable_normalize_unicode:
            self._transform_rules.append(lambda t: normalize_unicode(t, form=cfg.unicode_form))
        if cfg.enable_strip_modern_punctuation:
            self._transform_rules.append(strip_modern_punctuation)
        if cfg.enable_remove_layout_noise:
            self._transform_rules.append(remove_layout_noise)
        if cfg.enable_normalize_whitespace:
            self._transform_rules.append(normalize_whitespace)

        # 过滤规则（全部通过才保留）
        self._filter_rules.append(lambda t: filter_by_length(t, cfg.min_text_len, cfg.max_text_len))
        if cfg.enable_filter_non_chinese:
            self._filter_rules.append(lambda t: filter_by_cjk_ratio(t, cfg.min_cjk_ratio))

    def clean(self, input_path: str | Path, output_path: str | Path) -> CleaningStats:
        """执行清洗流程。

        Args:
            input_path: 输入 JSONL 路径（采集器的输出）
            output_path: 输出 JSONL 路径

        Returns:
            CleaningStats 清洗统计
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = CleaningStats()

        with (
            open(input_path, encoding="utf-8") as fin,
            open(output_path, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                stats.input_count += 1
                record = json.loads(line)
                text: str = record.get("text", "")
                stats.input_chars += len(text)

                # Phase 1: 转换
                for rule in self._transform_rules:
                    text = rule(text)

                if not text.strip():
                    continue

                # Phase 2: 过滤
                passed = True
                for frule in self._filter_rules:
                    if not frule(text):
                        passed = False
                        break

                if not passed:
                    if not filter_by_length(
                        text, self.config.min_text_len, self.config.max_text_len
                    ):
                        stats.filtered_by_length += 1
                    elif self.config.enable_filter_non_chinese and not filter_by_cjk_ratio(
                        text, self.config.min_cjk_ratio
                    ):
                        stats.filtered_by_cjk += 1
                    continue

                # 写出
                record["text"] = text
                stats.output_chars += len(text)
                stats.output_count += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "清洗完成: %d → %d 条 (%.1f%% 保留) | 字符: %d → %d",
            stats.input_count,
            stats.output_count,
            100 * stats.output_count / max(stats.input_count, 1),
            stats.input_chars,
            stats.output_chars,
        )
        return stats
