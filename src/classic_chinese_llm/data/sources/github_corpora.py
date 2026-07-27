"""GitHub 开源文言文语料适配器。

原始格式: 社区维护的文言文合集，通常为 txt 或 jsonl 格式。
"""

from __future__ import annotations

import json
from pathlib import Path

import chardet

from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.data.sources.base import BaseSource
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


class GitHubCorpusSource(BaseSource):
    """GitHub 开源文言文语料适配器。

    支持 txt（按空行切分文档）和 jsonl（每行一条记录）两种格式。
    """

    name = "github"
    display_name = "GitHub 开源语料"

    def discover(self, raw_dir: Path) -> list[Path]:
        """发现 txt 和 jsonl 文件。"""
        src_dir = raw_dir / self.name
        if not src_dir.exists():
            return []
        files: list[Path] = []
        files.extend(sorted(src_dir.glob("**/*.txt")))
        files.extend(sorted(src_dir.glob("**/*.jsonl")))
        files = [
            f for f in files if "readme" not in f.name.lower() and "license" not in f.name.lower()
        ]
        return files

    def parse(self, file_path: Path) -> list[SourceDocument]:
        """根据文件扩展名选择解析策略。"""
        if file_path.suffix == ".jsonl":
            return self._parse_jsonl(file_path)
        return self._parse_txt(file_path)

    def _parse_jsonl(self, file_path: Path) -> list[SourceDocument]:
        """解析 JSONL 格式语料。每行应包含 text 字段。"""
        docs: list[SourceDocument] = []
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("%s 第 %d 行 JSON 解析失败", file_path.name, line_num)
                    continue

                text = record.get("text") or record.get("content") or ""
                if not text or len(text) < 10:
                    continue

                docs.append(
                    SourceDocument(
                        text=text,
                        source=self.name,
                        title=record.get("title", ""),
                        author=record.get("author", ""),
                        era=record.get("era", ""),
                        genre=record.get("genre", ""),
                        metadata={"file": file_path.name, "line": line_num},
                    )
                )
        return docs

    def _parse_txt(self, file_path: Path) -> list[SourceDocument]:
        """解析 txt 格式语料。按空行分隔文档。"""
        raw_bytes = file_path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"

        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            text = raw_bytes.decode("utf-8", errors="replace")

        docs: list[SourceDocument] = []
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) >= 10:
                docs.append(
                    SourceDocument(
                        text=para,
                        source=self.name,
                        metadata={"file": file_path.name},
                    )
                )
        return docs
