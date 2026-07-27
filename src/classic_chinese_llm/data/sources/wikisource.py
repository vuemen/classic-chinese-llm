"""WikiSource 中文 XML dump 适配器。

原始格式: Wikimedia XML dump 文件，包含 <page> 元素。
使用 lxml iterparse 流式解析，避免全量加载大文件。
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.data.sources.base import BaseSource
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# XML namespace 通配符
_NS = "{http://www.mediawiki.org/xml/export-0.11/}"
_FALLBACK_NS = ""

# 非正文命名空间（模板、分类、文件等）
_SKIP_PREFIXES = ("Template:", "Category:", "File:", "Help:", "Wikipedia:", "Module:")


class WikiSourceSource(BaseSource):
    """WikiSource 中文 XML dump 适配器。

    使用 lxml.iterparse 流式解析 <page> 元素，
    提取 title 和 revision/text 内容，过滤非正文命名空间。
    """

    name = "wikisource"
    display_name = "WikiSource 中文"

    def discover(self, raw_dir: Path) -> list[Path]:
        """发现 XML dump 文件（支持 .xml 和 .xml.bz2 解压后的文件）。"""
        src_dir = raw_dir / self.name
        if not src_dir.exists():
            return []
        files = sorted(src_dir.glob("**/*.xml"))
        if not files:
            files = sorted(src_dir.glob("**/*.xml.bz2"))
        return files

    def parse(self, file_path: Path) -> list[SourceDocument]:
        """流式解析 WikiSource XML dump。

        处理 .xml.bz2 时需要先解压。对每个 <page>：
        - 跳过非正文命名空间（Template, Category 等）
        - 提取 title 和 revision 中的 text
        - elem.clear() 释放内存
        """
        # 如果是 bz2 压缩文件，先解压到临时位置
        if file_path.suffix == ".bz2":
            import bz2
            import tempfile

            logger.info("正在解压 %s ...", file_path.name)
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
                with bz2.open(file_path, "rb") as bz2_in:
                    while True:
                        chunk = bz2_in.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
                tmp_path = Path(tmp.name)
            try:
                return self._parse_xml(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

        return self._parse_xml(file_path)

    def _parse_xml(self, xml_path: Path) -> list[SourceDocument]:
        """解析解压后的 XML 文件。"""
        docs: list[SourceDocument] = []
        try:
            for _event, elem in etree.iterparse(
                str(xml_path),
                tag="page",
                huge_tree=True,
                recover=True,
            ):
                title = self._find_text(elem, "title")
                revision_text = self._find_revision_text(elem)

                if not title or not revision_text:
                    elem.clear()
                    continue

                # 跳过非正文命名空间
                if any(title.startswith(prefix) for prefix in _SKIP_PREFIXES):
                    elem.clear()
                    continue

                # 跳过太短的页面（重定向页通常只有 "#REDIRECT [[...]]"）
                if len(revision_text) < 50:
                    elem.clear()
                    continue

                docs.append(
                    SourceDocument(
                        text=revision_text,
                        source=self.name,
                        title=title,
                        metadata={"format": "wikitext"},
                    )
                )

                # 释放已处理元素的内存（关键优化）
                elem.clear()
                # 同时清理父节点的已处理子元素引用
                parent = elem.getparent()
                if parent is not None:
                    prev = elem.getprevious()
                    while prev is not None:
                        tmp = prev.getprevious()
                        parent.remove(prev)
                        prev = tmp

        except etree.XMLSyntaxError as e:
            logger.error("XML 解析错误: %s — %s", xml_path.name, e)

        return docs

    @staticmethod
    def _find_text(elem: etree._Element, tag: str) -> str:
        """在元素中查找子元素的文本（兼容有无 namespace）。"""
        child = elem.find(_NS + tag) or elem.find(tag)
        if child is not None and child.text:
            return str(child.text.strip())
        return ""

    def _find_revision_text(self, elem: etree._Element) -> str:
        """提取 revision 中的 text 内容。"""
        for rev in elem.iter("revision"):
            text_elem = rev.find(_NS + "text") or rev.find("text")
            if text_elem is not None and text_elem.text:
                text: str = str(text_elem.text).strip()
                # 过滤重定向页
                if text.upper().startswith("#REDIRECT"):
                    return ""
                return text
        return ""
