"""四库全书公开子集适配器。

原始格式: 开源镜像站获取的 txt 文件，通常按卷组织。
"""

from __future__ import annotations

import re
from pathlib import Path

import chardet

from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.data.sources.base import BaseSource
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# 卷标记正则
_VOLUME_PATTERN = re.compile(r"卷[第]?[之]?[一二三四五六七八九十百千\d]+")


class SiKuQuanShuSource(BaseSource):
    """四库全书公开子集适配器。

    解析 txt 文件：按卷标记或空行切分文档。
    四库全书按"经史子集"四部分类，优先从目录结构推断体裁。
    """

    name = "siku"
    display_name = "四库全书"

    # 目录名 → 体裁映射
    _GENRE_MAP = {
        "经": "经",
        "经部": "经",
        "史": "史",
        "史部": "史",
        "子": "子",
        "子部": "子",
        "集": "集",
        "集部": "集",
    }

    def discover(self, raw_dir: Path) -> list[Path]:
        """发现所有 txt 文件。"""
        src_dir = raw_dir / self.name
        if not src_dir.exists():
            return []
        return sorted(src_dir.glob("**/*.txt"))

    def parse(self, file_path: Path) -> list[SourceDocument]:
        """解析 txt 文件。从目录结构中推断体裁。"""
        raw_bytes = file_path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"

        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            text = raw_bytes.decode("utf-8", errors="replace")

        # 从目录结构推断体裁
        genre = self._infer_genre(file_path)

        docs: list[SourceDocument] = []
        # 按卷标记切分
        sections = _VOLUME_PATTERN.split(text)

        if len(sections) <= 1:
            # 无卷标记时按空行切分
            sections = text.split("\n\n")

        title = file_path.stem

        for section in sections:
            section = section.strip()
            if len(section) < 20:
                continue

            # 尝试从内容中提取卷名
            lines = section.split("\n")
            volume_title = title
            if lines:
                first = lines[0].strip()
                if _VOLUME_PATTERN.match(first) or len(first) < 30:
                    volume_title = f"{title}·{first}"
                    section = "\n".join(lines[1:]).strip()

            if section:
                docs.append(
                    SourceDocument(
                        text=section,
                        source=self.name,
                        title=volume_title,
                        genre=genre,
                        metadata={
                            "file": file_path.name,
                            "encoding": encoding,
                        },
                    )
                )

        return docs

    def _infer_genre(self, file_path: Path) -> str:
        """从文件路径的目录结构中推断四部分类。"""
        parts = file_path.parts
        for part in parts:
            category = self._GENRE_MAP.get(part)
            if category:
                return category
        return ""
