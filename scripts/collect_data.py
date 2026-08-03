"""数据采集 CLI 入口。

用法:
    python scripts/collect_data.py --raw-dir data/raw --output-dir data/processed
    python scripts/collect_data.py --sources daizhige
"""

from __future__ import annotations

import argparse
from pathlib import Path

from classic_chinese_llm.config.paths import PathConfig
from classic_chinese_llm.data.cleaner import Cleaner, CleanerConfig
from classic_chinese_llm.data.collector import Collector
from classic_chinese_llm.data.deduplicator import Deduplicator, DeduplicatorConfig
from classic_chinese_llm.data.formatter import Formatter, FormatterConfig
from classic_chinese_llm.data.sources.ctext import CtextSource
from classic_chinese_llm.data.sources.daizhige import DaiZhiGeSource
from classic_chinese_llm.data.sources.github_corpora import GitHubCorpusSource
from classic_chinese_llm.data.sources.sikuquanshu import SiKuQuanShuSource
from classic_chinese_llm.utils.logging_config import setup_logging


def _build_sources(raw_dir: Path) -> list:
    """构建所有数据源适配器实例。"""
    return [
        DaiZhiGeSource(raw_dir),
        GitHubCorpusSource(raw_dir),
        SiKuQuanShuSource(raw_dir),
        CtextSource(raw_dir),
    ]


_SOURCE_MAP = {
    s.name: s
    for s in [
        DaiZhiGeSource,
        GitHubCorpusSource,
        SiKuQuanShuSource,
        CtextSource,
    ]
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classical Chinese LLM — 数据管道（采集→清洗→去重→格式化）"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="原始数据根目录 (默认: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="处理后数据输出目录 (默认: data/processed)",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        help="指定采集的数据源（默认全部）",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="失败重试次数 (默认: 3)",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="跳过采集阶段，直接从已有 JSONL 开始",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="跳过清洗阶段",
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="跳过去重阶段",
    )
    parser.add_argument(
        "--skip-format",
        action="store_true",
        help="跳过格式化阶段",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=15000,
        help="最大 SFT 样本数 (默认: 15000)",
    )
    args = parser.parse_args()

    PathConfig.initialize(project_root=".")
    paths = PathConfig.get()
    setup_logging(level="INFO", log_file=str(paths.logs_dir / "collect_data.log"))

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    # ── 阶段 1: 采集 ──
    if not args.skip_collect:
        if args.sources:
            sources = [_SOURCE_MAP[name](raw_dir) for name in args.sources if name in _SOURCE_MAP]
        else:
            sources = _build_sources(raw_dir)

        collector = Collector(
            sources=sources,
            retry_attempts=args.retry,
        )
        collector.run(
            raw_dir=raw_dir,
            output_dir=output_dir,
        )

    collected_path = output_dir / "collected.jsonl"

    # ── 阶段 2: 清洗 ──
    cleaned_path = output_dir / "cleaned.jsonl"
    if not args.skip_clean:
        if not collected_path.exists():
            print(f"错误: 采集结果文件不存在: {collected_path}")
            return
        cleaner = Cleaner(CleanerConfig(min_text_len=20))
        cleaner.clean(collected_path, cleaned_path)
    else:
        cleaned_path = collected_path

    # ── 阶段 3: 去重 ──
    deduped_path = output_dir / "deduplicated.jsonl"
    if not args.skip_dedup:
        if not cleaned_path.exists():
            print(f"错误: 清洗结果文件不存在: {cleaned_path}")
            return
        dedup = Deduplicator(DeduplicatorConfig())
        dedup.deduplicate(cleaned_path, deduped_path)
    else:
        deduped_path = cleaned_path

    # ── 阶段 4: 格式化 ──
    if not args.skip_format:
        if not deduped_path.exists():
            print(f"错误: 去重结果文件不存在: {deduped_path}")
            return
        formatter = Formatter(FormatterConfig(max_samples=args.max_samples))
        instructions_dir = output_dir / "instructions"
        formatter.format(deduped_path, instructions_dir)


if __name__ == "__main__":
    main()
