"""数据管道模块。

Phase 2 数据管线:
    Collector → Cleaner → Deduplicator → Formatter

提供:
- Collector: 多数据源采集编排器
- Cleaner: 管道式文本清洗器
- Deduplicator: 两阶段去重器 (需要 datasketch)
- Formatter: 指令数据集格式化器
- SourceDocument: 统一文档模型
- BaseSource: 数据源适配器抽象基类

注意: 采用惰性导入 —— 仅在真正使用时才加载子模块。
这样测试 config/utils 等模块时不需要安装 datasketch/chardet 等可选依赖。
"""

from __future__ import annotations

from typing import Any

from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.data.sources.base import BaseSource


def __getattr__(name: str) -> Any:
    """惰性导入：仅在访问时才加载对应子模块。"""
    _imports: dict[str, tuple[str, str]] = {
        "Collector": ("classic_chinese_llm.data.collector", "Collector"),
        "Cleaner": ("classic_chinese_llm.data.cleaner", "Cleaner"),
        "CleanerConfig": ("classic_chinese_llm.data.cleaner", "CleanerConfig"),
        "CleaningStats": ("classic_chinese_llm.data.cleaner", "CleaningStats"),
        "Deduplicator": ("classic_chinese_llm.data.deduplicator", "Deduplicator"),
        "DeduplicatorConfig": ("classic_chinese_llm.data.deduplicator", "DeduplicatorConfig"),
        "DedupStats": ("classic_chinese_llm.data.deduplicator", "DedupStats"),
        "Formatter": ("classic_chinese_llm.data.formatter", "Formatter"),
        "FormatterConfig": ("classic_chinese_llm.data.formatter", "FormatterConfig"),
        "FormattingStats": ("classic_chinese_llm.data.formatter", "FormattingStats"),
        "TaskTemplate": ("classic_chinese_llm.data.formatter", "TaskTemplate"),
    }

    if name in _imports:
        module_path, attr = _imports[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 采集
    "Collector",
    "SourceDocument",
    "BaseSource",
    # 清洗
    "Cleaner",
    "CleanerConfig",
    "CleaningStats",
    # 去重
    "Deduplicator",
    "DeduplicatorConfig",
    "DedupStats",
    # 格式化
    "Formatter",
    "FormatterConfig",
    "FormattingStats",
    "TaskTemplate",
]
