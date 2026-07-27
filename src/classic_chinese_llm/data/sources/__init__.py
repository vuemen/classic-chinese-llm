"""数据源适配器模块。

提供:
- BaseSource: 抽象基类
- 5 个内置数据源适配器 (需要 chardet, lxml 等可选依赖)

注意: 采用惰性导入 —— 仅在真正访问具体适配器时才加载对应模块。
"""

from __future__ import annotations

from typing import Any

from classic_chinese_llm.data.sources.base import BaseSource


def __getattr__(name: str) -> Any:
    _imports: dict[str, tuple[str, str]] = {
        "CtextSource": ("classic_chinese_llm.data.sources.ctext", "CtextSource"),
        "DaiZhiGeSource": ("classic_chinese_llm.data.sources.daizhige", "DaiZhiGeSource"),
        "GitHubCorpusSource": (
            "classic_chinese_llm.data.sources.github_corpora",
            "GitHubCorpusSource",
        ),
        "SiKuQuanShuSource": (
            "classic_chinese_llm.data.sources.sikuquanshu",
            "SiKuQuanShuSource",
        ),
        "WikiSourceSource": (
            "classic_chinese_llm.data.sources.wikisource",
            "WikiSourceSource",
        ),
    }

    if name in _imports:
        import importlib

        module_path, attr = _imports[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseSource",
    "DaiZhiGeSource",
    "WikiSourceSource",
    "GitHubCorpusSource",
    "SiKuQuanShuSource",
    "CtextSource",
]
