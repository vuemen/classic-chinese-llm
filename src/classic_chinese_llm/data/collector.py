"""数据采集编排器。

遍历所有数据源，执行统一的 discover -> parse -> validate 流程，
输出统一 schema 的 JSONL。
"""

from __future__ import annotations

import time
from pathlib import Path

from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.data.sources.base import BaseSource
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


class Collector:
    """数据采集编排器。

    统一流程:
    1. 遍历已启用的 Source 列表
    2. 每个 Source: discover → 遍历返回的文件 → parse → validate
    3. 输出统一 JSONL 到 output_dir

    用法:
        sources = [DaiZhiGeSource(data_dir)]
        collector = Collector(sources, retry_attempts=3)
        collector.run(raw_dir=..., output_dir=...)
    """

    def __init__(
        self,
        sources: list[BaseSource],
        *,
        retry_attempts: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.sources = sources
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self._stats: dict[str, dict[str, int]] = {}

    def run(
        self,
        raw_dir: str | Path,
        output_dir: str | Path,
        *,
        output_filename: str = "collected.jsonl",
    ) -> Path:
        """执行全量采集，返回汇总 JSONL 路径。

        Args:
            raw_dir: 原始数据根目录
            output_dir: JSONL 输出目录
            output_filename: JSONL 文件名

        Returns:
            输出文件的 Path
        """
        raw_dir = Path(raw_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        total_docs = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            for source in self.sources:
                logger.info("处理数据源: %s (%s)", source.display_name, source.name)
                docs = self._collect_source(source, raw_dir)
                for doc in docs:
                    out_f.write(doc.to_jsonl_line() + "\n")
                total_docs += len(docs)

        self._print_stats()
        logger.info("采集完成 | 总计 %d 篇文档 → %s", total_docs, output_path)
        return output_path

    def _collect_source(
        self,
        source: BaseSource,
        raw_dir: Path,
    ) -> list[SourceDocument]:
        """采集单个数据源（含重试逻辑）。"""
        name = source.name

        # Phase 1: 发现文件
        files = source.discover(raw_dir)
        if not files:
            logger.warning("  %s: 无文件发现，跳过", source.display_name)
            self._stats[name] = {"files": 0, "docs": 0, "chars": 0}
            return []

        logger.info("  %s: 发现 %d 个文件", source.display_name, len(files))

        # Phase 2 & 3: 解析 + 校验
        all_docs: list[SourceDocument] = []
        for fp in files:
            parsed = self._parse_with_retry(source, fp)
            valid = [d for d in parsed if source.validate(d)]
            all_docs.extend(valid)

        # 后处理（子类可选）
        all_docs = source.post_process(all_docs)

        total_chars = sum(len(d.text) for d in all_docs)
        self._stats[name] = {
            "files": len(files),
            "docs": len(all_docs),
            "chars": total_chars,
        }
        logger.info(
            "  %s: 完成 → %d 篇 / %d 字符",
            source.display_name,
            len(all_docs),
            total_chars,
        )
        return all_docs

    def _parse_with_retry(
        self,
        source: BaseSource,
        file_path: Path,
    ) -> list[SourceDocument]:
        """带重试的解析单个文件。"""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return source.parse(file_path)
            except Exception:
                if attempt == self.retry_attempts:
                    logger.exception(
                        "  %s: 解析失败（重试 %d 次后跳过）: %s",
                        source.display_name,
                        self.retry_attempts,
                        file_path.name,
                    )
                else:
                    wait = self.retry_backoff**attempt
                    logger.warning(
                        "  %s: 重试 %d/%d（%.0fs 后退避）: %s",
                        source.display_name,
                        attempt,
                        self.retry_attempts,
                        wait,
                        file_path.name,
                    )
                    time.sleep(wait)
        return []

    def _print_stats(self) -> None:
        """打印采集统计报告。"""
        logger.info("=" * 50)
        logger.info("采集统计报告")
        logger.info("=" * 50)
        for name, stat in self._stats.items():
            chars_m = stat["chars"] / 1_000_000
            logger.info(
                "  %15s: %4d 文件 → %6d 篇 / %8.1fM 字符",
                name,
                stat["files"],
                stat["docs"],
                chars_m,
            )
