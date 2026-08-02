"""WikiSource 中文 XML dump 适配器。

原始格式: Wikimedia XML dump 文件，包含 <page> 元素。
使用 lxml iterparse 流式解析，避免全量加载大文件。

命名空间处理: 使用 {*} 通配符前缀匹配所有命名空间（包括无命名空间），
避免硬编码 Wikimedia export 版本号带来的兼容性问题。
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.data.sources.base import BaseSource
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# 非正文命名空间（模板、分类、文件等）—— title 前缀匹配
_SKIP_PREFIXES = ("Template:", "Category:", "File:", "Help:", "Wikipedia:", "Module:")


def _ns_tag(local_name: str) -> str:
    """生成命名空间通配符 tag，兼容任意或零命名空间。"""
    return f"{{*}}{local_name}"


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

        .xml.bz2 文件直接流式解压 → iterparse，不写临时文件，
        避免解压后 60-80GB 占用磁盘空间。
        """
        if file_path.suffix == ".bz2":
            import bz2

            logger.info("正在流式解压并解析 %s (不落盘)...", file_path.name)
            with bz2.open(file_path, "rb") as bz2_stream:
                return self._parse_xml(bz2_stream)

        return self._parse_xml(str(file_path))

    def _parse_xml(self, source: object) -> list[SourceDocument]:
        """流式解析 XML。

        Args:
            source: 文件路径字符串，或二进制文件对象（如 bz2 流）。
        """
        docs: list[SourceDocument] = []
        try:
            for _event, elem in etree.iterparse(
                source,
                tag=_ns_tag("page"),
                huge_tree=True,
                recover=True,
            ):
                title = self._find_child_text(elem, "title")
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
            logger.error("XML 解析错误: %s", e)

        return docs

    @staticmethod
    def _find_child_text(elem: etree._Element, tag: str) -> str:
        """查找子元素并返回其文本内容（命名空间无关）。"""
        child = elem.find(_ns_tag(tag))
        if child is not None and child.text:
            return str(child.text.strip())
        return ""

    def _find_revision_text(self, elem: etree._Element) -> str:
        """提取 revision 中的 text 内容（命名空间无关）。"""
        for rev in elem.iter(_ns_tag("revision")):
            text_elem = rev.find(_ns_tag("text"))
            if text_elem is not None and text_elem.text:
                text: str = str(text_elem.text).strip()
                # 过滤重定向页（仅取前缀判断，避免全文 upper() 导致 MemoryError）
                if text[:15].upper().startswith("#REDIRECT"):
                    return ""
                return text
        return ""
