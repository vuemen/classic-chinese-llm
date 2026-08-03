"""数据管道统一数据模型。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class SourceDocument:
    """统一文档模型 —— 所有数据源适配器的输出格式。

    字段:
        text: 正文内容（必填）
        source: 来源标识，如 daizhige, github, siku, ctext
        title: 篇名/书名
        author: 作者
        era: 朝代
        genre: 体裁
        url: 原始 URL
        chapter: 章节/卷
        metadata: 扩展元信息
        collected_at: 采集时间戳 (ISO 8601)
    """

    text: str
    source: str
    title: str = ""
    author: str = ""
    era: str = ""
    genre: str = ""
    url: str = ""
    chapter: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    collected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_jsonl_line(self) -> str:
        """序列化为一行 JSONL（ensure_ascii=False 保留中文字符）。"""
        return json.dumps(asdict(self), ensure_ascii=False)
