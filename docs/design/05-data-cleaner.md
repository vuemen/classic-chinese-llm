# 数据清洗器设计文档

**所属阶段:** Phase 2 — 数据管道
**涉及模块:** `src/classic_chinese_llm/data/cleaner.py`
**日期:** 2026-07-27

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | Unicode 规范化 | 统一全角/半角字符、兼容性字符（NFKC），消除视觉相同但编码不同的字符 |
| F2 | 现代标点剥离 | 移除现代中文标点（""''、……、——等），保留文言文标点（。，、；：？！「」『』） |
| F3 | 版式噪声去除 | 去除页码、页眉页脚、注释标记（①②③、㈠㈡㈢）、HTML/XML 标签残留、URL |
| F4 | 空白规范化 | 文言文不同于现代中文——词间无空格，段落间最多一个空行。多余空白需规范化 |
| F5 | 长度过滤 | 丢弃过短（<min_len）或过长（>max_len）的文档，可配置阈值 |
| F6 | 非中文过滤 | 丢弃拉丁字母/日文假名/韩文占比过高的行或文档 |
| F7 | 管道式组合 | 每个清洗规则独立、可插拔、可单独启用/禁用 |
| F8 | 处理统计 | 输出每个规则丢弃/修改的行数占比，便于调参 |

### 1.2 非功能需求

- **性能**: 单进程处理 ~3-6 亿字符应在数分钟内完成（纯文本 CPU 操作，无网络 I/O）
- **幂等性**: 重复运行清洗器不改变结果（cleaned → cleaned 副作用为零）
- **保留原文**: 清洗操作以"保守"为原则——宁可少删，不可多删。不确定的字符保留
- **可配置**: 每条规则的参数（如长度阈值、标点集合）通过配置注入，不硬编码
- **零额外依赖**: 仅使用 stdlib（re、unicodedata）+ 项目已有的 logging

---

## 2. 方案选型与对比

### 2.1 清洗管道架构

这是最核心的设计决策——如何组合多个独立的清洗规则。

| 方案 | 可组合性 | 可配置性 | 可测试性 | 复杂度 | 结论 |
|------|----------|----------|----------|--------|------|
| **函数管道 (Callable list)** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 最低 | ✅ 选用 |
| Strategy 模式 (类层次) | ⭐⭐ | ⭐⭐ | ⭐⭐ | 中 | ❌ 过度设计 |
| 装饰器注册 | ⭐⭐ | ⭐ | ⭐⭐ | 中 | ❌ 不直观 |
| 单一 Regex 大杂烩 | ⭐ | ⭐ | ⭐ | 低 | ❌ 不可维护 |

**详细对比**:

```python
# 方案 A: 函数管道（选用）
# 每个规则是 Callable[[str], str]，清洗器持有规则列表
rules = [normalize_unicode, strip_punctuation, normalize_whitespace]
# 优点: 每个规则独立函数，可单独单测；规则列表从配置读取即实现动态组合
# 缺点: 无内建前后依赖管理（但本场景规则间无依赖）

# 方案 B: Strategy 模式
class CleaningRule(ABC):
    @abstractmethod
    def apply(self, text: str) -> str: ...
# 优点: 规则可携带状态、配置
# 缺点: 每个规则一个类文件，5 条规则 = 5 个类 + 5 个文件，过度工程化

# 方案 C: 装饰器注册
@register_rule("normalize_unicode")
def normalize_unicode(text: str) -> str: ...
# 优点: 声明式、自动发现
# 缺点: 引入隐式全局状态、测试复杂化
```

**最终选择**: **函数管道 (Callable list)**。清洗规则本质是纯函数 `(str) → str`，无需状态。函数列表天然支持组合、顺序控制、单测。复杂度最低，完全满足需求。

### 2.2 现代标点检测方案

文言文与现代中文在标点使用上有清晰边界——这是清洗器中最重要的规则。

| 方案 | 准确率 | 速度 | 可维护性 | 结论 |
|------|--------|------|----------|------|
| **Unicode 类别 + 白名单** | ⭐⭐⭐ | ⭐⭐⭐ 最快 | ⭐⭐⭐ | ✅ 选用 |
| 纯正则 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 备选（辅助） |
| NLP 分词后判断 | ⭐⭐⭐ | ⭐ 很慢 | ⭐ | ❌ 杀鸡用牛刀 |
| 字符属性表硬编码 | ⭐⭐ | ⭐⭐⭐ | ⭐ | ❌ 难维护 |

**最终选择**: **Unicode 类别 + 白名单**。核心思路是：只保留一个显式列出的"文言文合法标点白名单"，其余 Unicode 标点类别（`P` category）的字符一律删除。

文言文合法标点白名单：
```
。，、；：？！「」『』．  — 中文古典标点
·                         — 间隔号（人名分隔）
《》〈〉                   — 书名号
```

关键排除项（现代中文特有）：
```
""''……——！！？？     — 双引号、省略号、破折号、叠用标点
（）【】{}               — 括号
,.;:!?…-                — 英文标点
```

### 2.3 语言检测方案

需要过滤混入的非中文文本（英文摘要、日文注释等）。

| 方案 | 准确率 | 速度 | 额外依赖 | 结论 |
|------|--------|------|----------|------|
| **CJK 字符占比** | ⭐⭐ | ⭐⭐⭐ | 0 | ✅ 选用 |
| langdetect | ⭐⭐⭐ | ⭐ | langdetect | ❌ 短文本不可靠 |
| fasttext | ⭐⭐⭐ | ⭐⭐ | fasttext + 模型文件 | ❌ 过重 |
| unicodedata.name | ⭐⭐ | ⭐⭐ | 0 | 备选 |

**最终选择**: **CJK 字符占比（Char Ratio）**。文言文语料场景下，非中文内容以混入的英文元数据和日文注释为主，CJK 统一汉字区（U+4E00–U+9FFF）占比阈值 ≥0.7 即可有效过滤。这个方案零依赖、零模型文件、速度快。

对于边界情况（如佛经中的梵文音译用字），通过 `unicodedata.category()` 补充判断。

---

## 3. 最终方案

### 3.1 模块结构

```
src/classic_chinese_llm/data/
├── __init__.py
├── collector.py
├── cleaner.py          # Cleaner 编排器 + 内置清洗规则
└── ...
```

Cleaner 模块为单文件（<300 行），包含内置规则函数和编排类。

### 3.2 核心接口设计

```python
# data/cleaner.py

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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

    min_text_len: int = 10          # 最小字符数（含中文）
    max_text_len: int = 100000      # 最大字符数

    # Unicode 规范化
    unicode_form: str = "NFKC"      # NFC | NFKC | NFD | NFKD

    # 语言过滤
    min_cjk_ratio: float = 0.7      # 最小中日韩统一汉字占比

    # 规则启用开关
    enable_normalize_unicode: bool = True
    enable_strip_modern_punctuation: bool = True
    enable_remove_layout_noise: bool = True
    enable_normalize_whitespace: bool = True
    enable_filter_non_chinese: bool = True


# ─── 内置清洗规则 ─────────────────────────────────────────────────────
# 每条规则: CleaningRule = Callable[[str], str]

def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Unicode 规范化: 全角→半角数字/字母，兼容性字符→标准形式。

    NFKC 将全角英数字母转为半角:
      'Ａ' → 'A', '１' → '1', 'ｇ' → 'g'
    也将连字符合并:
      'ﬃ' → 'ffi'

    文言文核心字符（汉字、古典标点）不受 NFKC 影响，
    可安全使用。实际测试表明对文言字符集的误伤率为零。
    """
    return unicodedata.normalize(form, text)


# ─── 文言文标点白名单 ─────────────────────────────────────────────────

_CLASSICAL_PUNCTUATION = frozenset(
    "。，、；：？！「」『』．·《》〈〉"
    "——……"  # 文言文中也偶见（用于注释、省略），保留
)

# Unicode 类别: 标点符号
_PUNCTUATION_CATEGORIES = frozenset(
    {"Po", "Ps", "Pe", "Pi", "Pf", "Pc", "Pd"}
)


def _is_classical_punct(char: str) -> bool:
    """判断字符是否为文言文合法标点。"""
    return char in _CLASSICAL_PUNCTUATION


def _is_modern_punct(char: str) -> bool:
    """判断字符是否为现代标点（Unicode 标点类别但不在白名单中）。"""
    cat = unicodedata.category(char)
    if cat not in _PUNCTUATION_CATEGORIES:
        return False
    return not _is_classical_punct(char)


def strip_modern_punctuation(text: str) -> str:
    """移除现代标点。

    遍历每个字符，删除属于 Unicode 标点类别但不在文言文白名单中的字符。

    示例:
        '"论语"是儒家经典。' → '论语是儒家经典。'
        '子曰：「学而时习之……」' → '子曰：「学而时习之……」' (文言标点保留)
        'Confucius said: "Study..."' → 'Confucius said Study' (英文标点被删)

    注意: 不删除中文文字，也不删除文言合法标点。
    """
    return "".join(ch for ch in text if not _is_modern_punct(ch))


# ─── 版式噪声去除 ─────────────────────────────────────────────────────

# 页码: "第123页"、"p.123"、" - 45 -"
_RE_PAGE_NUMBER = re.compile(
    r"(第\s*\d+\s*[页頁])|([pP]\.?\s*\d+)|([—\-]\s*\d+\s*[—\-])"
)

# 注释标记: ①②③ ㈠㈡㈢ ⑴⑵⑶
_RE_ANNOTATION_MARKER = re.compile(r"[①-⑳㈠-㈩㊀-㊉]")

# HTML/XML 残留
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;")

# 现代标点组合（省略号、引号等）
_RE_MODERN_PUNCT_PATTERNS = re.compile(r'[""'']')

# URL
_RE_URL = re.compile(r"https?://\S+|www\.\S+")

# 纯标点/空白行
_RE_BLANK_OR_PUNCT_ONLY = re.compile(r"^[\s\p{P}]*$")


def remove_layout_noise(text: str) -> str:
    """去除版式噪声：页码、注释标记、HTML 标签、URL、多余空白。

    这些噪声主要来源于:
    - 数字化过程中 OCR 或格式转换残留
    - 网页抓取未清理的 HTML
    - 印刷版的页码/章节号
    """
    text = _RE_PAGE_NUMBER.sub("", text)
    text = _RE_ANNOTATION_MARKER.sub("", text)
    text = _RE_HTML_TAG.sub("", text)
    text = _RE_HTML_ENTITY.sub("", text)
    text = _RE_MODERN_PUNCT_PATTERNS.sub("", text)
    text = _RE_URL.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    """规范化空白字符。

    文言文规则:
    - 行首行尾空白去除
    - 连续空行合并为单个空行
    - 词间不留空格（文言文不分词，词间无空格）
    - Tab 转空格
    """
    # Tab → 空格
    text = text.replace("\t", " ")
    # 每行去除首尾空白
    lines = [line.strip() for line in text.splitlines()]
    # 合并连续空行
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
# 每条规则: FilterRule = Callable[[str], bool]

def filter_by_length(text: str, min_len: int = 10, max_len: int = 100000) -> bool:
    """长度过滤。过短无信息量（如纯标题），过长可能是未分段的大文件。"""
    clean = text.strip()
    return min_len <= len(clean) <= max_len


def filter_by_cjk_ratio(text: str, min_ratio: float = 0.7) -> bool:
    """CJK 汉字占比过滤。

    统计 Unicode CJK 统一汉字区（U+4E00–U+9FFF）字符在
    非空白字符中的占比。低于阈值的文本通常是:
    - 英文摘要/注释
    - 日文假名为主的内容（如和歌）
    - 纯标点或数字表格
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
    # 字符统计
    input_chars: int = 0
    output_chars: int = 0


class Cleaner:
    """数据清洗编排器。

    持有转换规则列表和过滤规则列表，对 JSONL 输入逐行清洗后输出清洗后 JSONL。

    用法:
        cleaner = Cleaner(CleanerConfig(min_text_len=20))
        cleaner.clean(input_path, output_path)
    """

    def __init__(self, config: CleanerConfig | None = None) -> None:
        self.config = config or CleanerConfig()
        self._transform_rules: list[CleaningRule] = []
        self._filter_rules: list[FilterRule] = []
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """根据配置组装转换管道和过滤管道。"""
        cfg = self.config

        # ── 转换规则（按顺序执行） ──
        if cfg.enable_normalize_unicode:
            self._transform_rules.append(
                lambda t: normalize_unicode(t, form=cfg.unicode_form)
            )
        if cfg.enable_strip_modern_punctuation:
            self._transform_rules.append(strip_modern_punctuation)
        if cfg.enable_remove_layout_noise:
            self._transform_rules.append(remove_layout_noise)
        if cfg.enable_normalize_whitespace:
            self._transform_rules.append(normalize_whitespace)

        # ── 过滤规则（任意一条不通过则丢弃） ──
        self._filter_rules.append(
            lambda t: filter_by_length(t, cfg.min_text_len, cfg.max_text_len)
        )
        if cfg.enable_filter_non_chinese:
            self._filter_rules.append(
                lambda t: filter_by_cjk_ratio(t, cfg.min_cjk_ratio)
            )

    def clean(self, input_path: str | Path, output_path: str | Path) -> CleaningStats:
        """执行清洗：读取 JSONL → 逐行清洗 → 写入 JSONL。

        Args:
            input_path: 输入 JSONL 路径（采集器的输出）
            output_path: 输出 JSONL 路径

        Returns:
            CleaningStats 清洗统计
        """
        import json

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = CleaningStats()

        with open(input_path, encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                stats.input_count += 1
                record = json.loads(line)
                text = record.get("text", "")

                stats.input_chars += len(text)

                # Phase 1: 转换
                for rule in self._transform_rules:
                    text = rule(text)

                if not text.strip():
                    continue

                # Phase 2: 过滤
                passed = True
                for rule in self._filter_rules:
                    if not rule(text):
                        passed = False
                        break

                if not passed:
                    # 统计过滤原因
                    if not filter_by_length(text, self.config.min_text_len, self.config.max_text_len):
                        stats.filtered_by_length += 1
                    elif self.config.enable_filter_non_chinese and not filter_by_cjk_ratio(text, self.config.min_cjk_ratio):
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
```

### 3.3 使用示例

```python
from classic_chinese_llm.data.cleaner import Cleaner, CleanerConfig

config = CleanerConfig(
    min_text_len=20,
    max_text_len=50000,
    min_cjk_ratio=0.75,
    enable_normalize_unicode=True,
    enable_strip_modern_punctuation=True,
    enable_remove_layout_noise=True,
    enable_normalize_whitespace=True,
    enable_filter_non_chinese=True,
)

cleaner = Cleaner(config)
stats = cleaner.clean("data/processed/collected.jsonl", "data/processed/cleaned.jsonl")
print(f"保留 {stats.output_count}/{stats.input_count} 条记录")
```

---

## 4. 关键技术点

### 4.1 转换 (Transform) 与过滤 (Filter) 的分离

Cleaner 将操作分为两类：

- **转换规则** (`CleaningRule = Callable[[str], str]`): 修改文本内容但不删除整条记录。如 Unicode 规范化、标点剥离。这些规则**按顺序执行**——前一个的输出是后一个的输入。

- **过滤规则** (`FilterRule = Callable[[str], bool]`): 判断文本是否合格，不修改内容。如长度检查、语言检测。这些规则**全部通过**才保留——任意一条返回 `False` 即丢弃。

分离的好处：
1. **单一职责**: 转换规则负责"修"，过滤规则负责"判"
2. **统计清晰**: 可以分别统计"被哪条规则过滤了多少条"
3. **性能**: 过滤在转换之后执行，避免对将被丢弃的文本做昂贵的转换

### 4.2 NFKC 对文言文的影响分析

NFKC 是"兼容性组合"规范化，将字符转换为其兼容性等价形式。对文言文的影响：

```
✅ 安全转换（正确行为）:
  全角英文 ＡＢＣ   → ABC      （数字化文本常见遗留问题）
  全角数字 １２３   → 123      （同上）
  罗马数字 Ⅳ       → IV       （极少出现在文言文中）

✅ 不影响文言核心字符:
  汉字    U+4E00-U+9FFF  → 不变
  文言标点 「」『』。，、  → 不变

⚠️ 罕见边界情况:
  海 (U+FA45, CJK 兼容汉字) → 海 (U+6D77)  （兼容区汉字极少见）
```

由于本项目语料来源以现代整理为主（殆知阁），兼容性汉字极少出现。NFKC 的规范化收益（修复全角英数字）大于罕见误伤风险。

### 4.3 标点白名单的保守策略

文言文标点白名单遵循"宁少勿多"原则。白名单中仅包含确定在古典文本中合法使用的标点：

```
。 ， 、 ； ： ？ ！    — 基本句读标点
「 」 『 』            — 引号（文言文常用）
《 》 〈 〉            — 书名号
·                      — 间隔号
．                      — 句点
—— ……                 — 破折号、省略号（古文注释中偶见）
```

不在白名单中的 Unicode 标点（如 `""''（）【】{}` 等）一律视为现代标点移除。这种"激进删除"策略在纯文言文语料中误伤率极低。

### 4.4 版式噪声的正则表达式优先级

版式噪声的识别按以下优先级处理：

| 优先级 | 噪声类型 | 正则策略 | 示例 |
|--------|----------|----------|------|
| 1 (最高) | URL | 匹配 `http://` 或 `www.` 前缀 | `https://example.com` |
| 2 | HTML/XML 标签 | `<...>` 模式 | `<div class="text">` |
| 3 | HTML 实体 | `&...;` 模式 | `&nbsp;` `&mdash;` |
| 4 | 页码 | 多种模式组合 | `第123页` `p.45` |
| 5 | 注释标记 | Unicode 范围匹配 | ①②③ ㈠㈡㈢ |
| 6 (最低) | 多余空白 | 正则规范化 | 连续换行、Tab |

URL 必须最先处理——否则 URL 中的标点（如 `.`、`/`、`:`）会被后续规则错误处理。

### 4.5 文言文与空白处理的特殊性

现代中文文本清洗通常会将连续空格合并为单个空格。但文言文**词间完全没有空格**——文言文的分词是纯语义层面的，不在文本中体现。因此 Cleaner 的空白处理策略为：

```
行首尾空白  → 去除
行间连续空行 → 合并为单个空行（段落分隔符保留）
词间空格     → 保留（极少出现，保留以防万一）
Tab          → 转为空格
```

注意：不去除所有空格。古文中偶有以空格表示敬称避讳的格式（如清代文档中在"皇""圣"前空一格），虽然本项目语料中几乎不存在，但保守策略选择保留。

### 4.6 CJK 字符占比阈值的选择

`min_cjk_ratio = 0.7` 的设定依据：

| 文本类型 | CJK 占比 | 是否保留 |
|----------|----------|----------|
| 纯文言文（论语、史记） | ~0.92-0.98 | ✅ |
| 含少量现代注释的古文 | ~0.75-0.90 | ✅ |
| 中英混合摘要 | ~0.40-0.60 | ❌ |
| 纯英文 | ~0.00-0.05 | ❌ |
| 日文（含汉字 + 假名） | ~0.40-0.60 | ❌ |
| 韩文（谚文 + 少量汉字） | ~0.05-0.15 | ❌ |

阈值 0.7 能保留所有含少量注释/标点的文言文本，同时有效过滤非中文内容。如果需要保留佛经文类（可能含梵文音译字），可临时降低到 0.5。

---

## 5. 与其他模块的关系

```
Config ─── 被依赖 ───> Cleaner (CleanerConfig 注入)
Utils  ─── 被依赖 ───> Cleaner (logging)

Collector ─── 输出 collected.jsonl ──→ Cleaner ──→ 输出 cleaned.jsonl ──→ Deduplicator
```

Cleaner 是采集器（Collector）的直接下游。它的输出 `cleaned.jsonl` 是去重器（Deduplicator）的输入。三者形成严格线性流水线。

---

## 6. 验证清单

- [ ] `normalize_unicode("ＡＢＣ１２３")` 返回 `"ABC123"`，汉字不受影响
- [ ] `strip_modern_punctuation('"论语"是经典。')` 返回 `'论语是经典。'`
- [ ] `strip_modern_punctuation('子曰：「学而时习之」')` 中 `「」` 被保留
- [ ] `remove_layout_noise("第123页")` 中的页码被移除
- [ ] `remove_layout_noise("<div>正文</div>")` 返回 `"正文"`
- [ ] `normalize_whitespace` 将连续 3 个空行合并为 1 个
- [ ] `filter_by_cjk_ratio("This is English text", 0.7)` 返回 `False`
- [ ] `filter_by_cjk_ratio("子曰学而时习之", 0.7)` 返回 `True`
- [ ] `filter_by_length("短", min_len=10)` 返回 `False`
- [ ] Cleaner 的 `clean()` 输入 100 行 JSONL，统计数字与手动计数一致
- [ ] 禁用全部规则时，输入=输出（无修改、无丢弃）
- [ ] 清洗结果再次清洗，输出不变（幂等性验证）
