"""ctext.org 手动补充适配器。

原始格式: 从 ctext.org 手动下载的 txt 文件，质量最高但无批量接口。
"""

from __future__ import annotations

from pathlib import Path

import chardet

from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.data.sources.base import BaseSource
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


class CtextSource(BaseSource):
    """ctext.org 手动补充适配器。

    处理手动下载的高质量典籍 txt 文件。
    ctext.org 文件通常为 UTF-8 编码，按章节组织。
    """

    name = "ctext"
    display_name = "ctext.org"

    def discover(self, raw_dir: Path) -> list[Path]:
        """发现所有 txt 文件。"""
        src_dir = raw_dir / self.name
        if not src_dir.exists():
            return []
        return sorted(src_dir.glob("**/*.txt"))

    def parse(self, file_path: Path) -> list[SourceDocument]:
        """解析 ctext txt 文件。

        ctext 文件通常有较好的结构：文件名即书名，每行为一段。
        """
        raw_bytes = file_path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"

        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            text = raw_bytes.decode("utf-8", errors="replace")

        title = file_path.stem

        docs: list[SourceDocument] = []
        # 按空行切分，保持章节完整性
        sections = text.split("\n\n")

        for section in sections:
            section = section.strip()
            if len(section) < 10:
                continue

            docs.append(
                SourceDocument(
                    text=section,
                    source=self.name,
                    title=title,
                    metadata={
                        "file": file_path.name,
                        "encoding": encoding,
                    },
                )
            )

        return docs
