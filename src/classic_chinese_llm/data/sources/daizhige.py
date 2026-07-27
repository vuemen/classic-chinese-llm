"""殆知阁古代汉语语料库适配器。

原始格式: 打包的 txt 文件，每行为一篇文档或按空行分隔。
常见结构: 【篇名】正文内容 或 首行为标题。
"""

from __future__ import annotations

import re
from pathlib import Path

import chardet

from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.data.sources.base import BaseSource
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# 篇名标记正则（适配多种格式）
_TITLE_PATTERN = re.compile(r"^[【\[《](.+?)[】\]》]\s*")


class DaiZhiGeSource(BaseSource):
    """殆知阁古代汉语语料库适配器。

    解析 txt 文件：自动检测编码，按空行或标题标记切分文档。
    """

    name = "daizhige"
    display_name = "殆知阁"

    def discover(self, raw_dir: Path) -> list[Path]:
        """发现所有 txt 文件，排除 README 和说明文件。"""
        src_dir = raw_dir / self.name
        if not src_dir.exists():
            return []
        files = sorted(src_dir.glob("**/*.txt"))
        files = [f for f in files if "readme" not in f.name.lower() and "说明" not in f.name]
        return files

    def parse(self, file_path: Path) -> list[SourceDocument]:
        """解析 txt 文件：自动检测编码，按空行切分文档。"""
        raw_bytes = file_path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"

        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            logger.warning("编码检测失败: %s, fallback UTF-8", file_path.name)
            text = raw_bytes.decode("utf-8", errors="replace")

        docs: list[SourceDocument] = []
        paragraphs = re.split(r"\n\s*\n", text)

        for para in paragraphs:
            para = para.strip()
            if len(para) < 10:
                continue

            title = ""
            match = _TITLE_PATTERN.match(para)
            if match:
                title = match.group(1).strip()
                body = para[match.end() :].strip()
            else:
                first_line = para.split("\n")[0].strip()
                if len(first_line) < 50 and not first_line.endswith(("。", "，", "；")):
                    title = first_line
                    body = para[len(first_line) :].strip()
                else:
                    body = para

            if body:
                docs.append(
                    SourceDocument(
                        text=body,
                        source=self.name,
                        title=title,
                        metadata={
                            "file": file_path.name,
                            "encoding": encoding,
                        },
                    )
                )

        return docs
