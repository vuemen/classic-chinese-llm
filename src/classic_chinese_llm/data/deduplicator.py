"""数据去重器。

两阶段去重:
1. SHA-256 精确去重（字节级完全相同的文档）
2. MinHash + LSH 近似去重（Jaccard 相似度 ≥ threshold 的文档对）

性能优化:
- MinHash 构建通过 multiprocessing 并行化
- shingle 生成用正则替代 O(n²) 的 while 空格压缩
- Jaccard 计算前用长度比过滤明显不相似的候选对
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


# ─── 配置 ──────────────────────────────────────────────────────────────


@dataclass
class DeduplicatorConfig:
    """去重器参数配置。"""

    num_perm: int = 128
    shingle_size: int = 5
    jaccard_threshold: float = 0.85
    lsh_num_bands: int | None = None  # None = datasketch 自动选择
    lsh_num_rows: int | None = None  # None = datasketch 自动选择
    seed: int = 42
    keep_strategy: str = "longest"  # "longest" | "earliest" | "most_complete"
    enable_exact_dedup: bool = True
    enable_approx_dedup: bool = True
    n_jobs: int = 0  # 0 = 自动 (Windows 安全值 min(cpu_count, 4)), 1 = 禁用并行


# ─── Shingle 生成 ──────────────────────────────────────────────────────

# 空白字符压缩：一步到位，避免 O(n²) 的 while 循环
_RE_WHITESPACE = re.compile(r"\s+")


def _char_shingles(text: str, k: int = 5) -> set[str]:
    """生成字符级 k-gram shingle 集合。"""
    clean = _RE_WHITESPACE.sub(" ", text)
    if len(clean) < k:
        return set()
    return {clean[i : i + k] for i in range(len(clean) - k + 1)}


# ─── 精确去重 ──────────────────────────────────────────────────────────


def _compute_sha256(text: str) -> str:
    """计算文本的 SHA-256 哈希值。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _exact_dedup(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """精确去重：相同 SHA-256 → 保留第一条。"""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0

    for record in records:
        h = _compute_sha256(record.get("text", ""))
        if h in seen:
            removed += 1
        else:
            seen.add(h)
            deduped.append(record)

    return deduped, removed


# ─── 近似去重 ──────────────────────────────────────────────────────────


def _build_minhashes_parallel(
    records: list[dict[str, Any]],
    indices: list[int],
    config: DeduplicatorConfig,
    n_jobs: int,
) -> dict[int, MinHash]:
    """并行构建所有文档的 MinHash 签名。

    使用独立的 _minhash_worker 模块避免 Windows spawn 模式下
    每个子进程加载 torch 等重型依赖（~2GB/进程）。
    """
    from classic_chinese_llm.data._minhash_worker import build_single_minhash

    # 获取参考 MinHash 的 scheme（datasketch 2.0+ 重构时需要指定）
    ref = MinHash(num_perm=config.num_perm, seed=config.seed)
    scheme = ref.scheme

    tasks: list[tuple[int, str, int, int, int]] = []
    for idx in indices:
        text = records[idx].get("text", "")
        tasks.append((idx, text, config.num_perm, config.seed, config.shingle_size))

    minhashes: dict[int, MinHash] = {}

    logger.info("  启动 %d 个 worker 进程 (独立模块, 低内存占用)", n_jobs)
    with mp.Pool(processes=n_jobs) as pool:
        # 使用 imap_unordered 获得进度反馈
        results = pool.imap_unordered(build_single_minhash, tasks, chunksize=500)
        for i, (idx, digest) in enumerate(results):
            m = MinHash(
                num_perm=config.num_perm,
                seed=config.seed,
                hashvalues=digest,
                scheme=scheme,
            )
            minhashes[idx] = m
            if (i + 1) % 200_000 == 0:
                logger.info("  已构建 %d / %d 个 MinHash 签名", i + 1, len(indices))

    return minhashes


def _length_ratio(a_len: int, b_len: int) -> float:
    """计算两个字符串的长度比（较小/较大，0~1）。"""
    if a_len == 0 or b_len == 0:
        return 0.0
    return min(a_len, b_len) / max(a_len, b_len)


def _approx_dedup(
    records: list[dict[str, Any]],
    config: DeduplicatorConfig,
) -> tuple[list[dict[str, Any]], int]:
    """近似去重：MinHash + LSH 检测相似文档对，按连通分量分组去重。"""
    num_perm = config.num_perm
    threshold = config.jaccard_threshold
    # Windows spawn 模式下每个子进程会加载 torch DLL (~2GB 虚拟内存)
    # 保守默认 4 个 worker，用户可通过 config.n_jobs 覆盖
    if config.n_jobs > 0:
        n_jobs = config.n_jobs
    else:
        n_jobs = min(mp.cpu_count(), 4)

    # 跳过低信息量文档（<30 字符），它们不参与 LSH 索引
    valid_indices = [i for i, r in enumerate(records) if len(r.get("text", "")) >= 30]
    short_indices = [i for i, r in enumerate(records) if len(r.get("text", "")) < 30]

    if not valid_indices:
        return records, 0

    logger.info("  有效文档: %d, 短文档(跳过): %d", len(valid_indices), len(short_indices))

    # ── 并行构建 MinHash ──
    if n_jobs > 1:
        logger.info("  并行构建 MinHash (n_jobs=%d)...", n_jobs)
        minhashes = _build_minhashes_parallel(records, valid_indices, config, n_jobs)
    else:
        logger.info("  串行构建 MinHash...")
        minhashes = {}
        for i, idx in enumerate(valid_indices):
            m = MinHash(num_perm=num_perm, seed=config.seed)
            shingles = _char_shingles(records[idx].get("text", ""), k=config.shingle_size)
            if not shingles:
                m.update(b"")
            else:
                for s in shingles:
                    m.update(s.encode("utf-8"))
            minhashes[idx] = m
            if (i + 1) % 200_000 == 0:
                logger.info("  已构建 %d / %d 个 MinHash 签名", i + 1, len(valid_indices))

    logger.info("  MinHash 签名构建完成")

    # ── 构建 LSH 索引 ──
    logger.info("  构建 LSH 索引...")
    lsh_params = None
    if config.lsh_num_bands is not None and config.lsh_num_rows is not None:
        lsh_params = (config.lsh_num_bands, config.lsh_num_rows)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, params=lsh_params)

    for idx in valid_indices:
        lsh.insert(idx, minhashes[idx])

    # ── 查询 LSH，构建相似图 ──
    logger.info("  LSH 查询，构建相似图...")
    # 预缓存文档长度，用于快速过滤
    doc_lengths = {idx: len(records[idx].get("text", "")) for idx in valid_indices}

    graph: dict[int, set[int]] = defaultdict(set)
    pairs_checked = 0
    pairs_skipped = 0

    for i, idx in enumerate(valid_indices):
        candidates = lsh.query(minhashes[idx])
        idx_len = doc_lengths[idx]
        for cand in candidates:
            if cand == idx:
                continue
            if cand in graph.get(idx, set()):
                continue  # 已通过双向添加记录过

            # 长度预过滤：长度比 < 0.25 的两个文档不可能 Jaccard ≥ 0.85
            # 推导: Jaccard(A,B) = |A∩B| / |A∪B| ≥ 0.85
            #   → |A∩B| ≥ 0.85 × |A∪B| ≥ 0.85 × max(|A|, |B|)
            #   → min(|A|, |B|) ≥ |A∩B| ≥ 0.85 × max(|A|, |B|)
            #   → min/max ≥ 0.85（字符层面近似，shingle 层面用更宽松的阈值）
            cand_len = doc_lengths[cand]
            if _length_ratio(idx_len, cand_len) < 0.25:
                pairs_skipped += 1
                continue

            actual_jaccard = minhashes[idx].jaccard(minhashes[cand])
            pairs_checked += 1
            if actual_jaccard >= threshold:
                graph[idx].add(cand)
                graph[cand].add(idx)

        if (i + 1) % 500_000 == 0:
            logger.info(
                "  已查询 %d / %d, 检验 %d 对, 跳过 %d 对",
                i + 1,
                len(valid_indices),
                pairs_checked,
                pairs_skipped,
            )

    logger.info("  图构建完成: %d 个节点, %d 条边", len(graph), sum(len(v) for v in graph.values()))

    # ── Union-Find 合并连通分量 ──
    parent = {i: i for i in valid_indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a in graph:
        for b in graph[a]:
            union(a, b)

    # ── 收集分组 ──
    groups_map: dict[int, list[int]] = defaultdict(list)
    for idx in valid_indices:
        groups_map[find(idx)].append(idx)

    # ── 每组保留一条 ──
    deduped: list[dict[str, Any]] = []
    removed = 0
    kept_set: set[int] = set()

    for group in groups_map.values():
        if len(group) == 1:
            deduped.append(records[group[0]])
            kept_set.add(group[0])
        else:
            keeper = _select_keeper(group, records, config.keep_strategy)
            deduped.append(records[keeper])
            kept_set.add(keeper)
            removed += len(group) - 1

    # 添加短文档（不参与近似去重但参与精确去重）
    for idx in short_indices:
        deduped.append(records[idx])

    return deduped, removed


def _select_keeper(
    group: list[int],
    records: list[dict[str, Any]],
    strategy: str,
) -> int:
    """从重复文档组中选择保留哪一条。"""
    if strategy == "earliest":
        return min(group)
    if strategy == "most_complete":
        return max(group, key=lambda i: sum(1 for v in records[i].values() if v))
    # "longest" (default)
    return max(group, key=lambda i: len(records[i].get("text", "")))


# ─── 去重编排器 ────────────────────────────────────────────────────────


@dataclass
class DedupStats:
    """去重统计信息。"""

    input_count: int = 0
    after_exact: int = 0
    exact_removed: int = 0
    after_approx: int = 0
    approx_removed: int = 0
    input_chars: int = 0
    output_chars: int = 0


class Deduplicator:
    """两阶段去重编排器。

    阶段 1: SHA-256 精确去重
    阶段 2: MinHash + LSH 近似去重（并行 MinHash 构建 + 长度预过滤）

    用法:
        dedup = Deduplicator(DeduplicatorConfig())
        stats = dedup.deduplicate(input_path, output_path)
    """

    def __init__(self, config: DeduplicatorConfig | None = None) -> None:
        self.config = config or DeduplicatorConfig()

    def deduplicate(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> DedupStats:
        """执行两阶段去重，输出去重后 JSONL。

        Args:
            input_path: 输入 JSONL 路径（Cleaner 的输出）
            output_path: 输出 JSONL 路径

        Returns:
            DedupStats 统计信息
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = DedupStats()

        # 加载全部记录到内存
        with open(input_path, encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

        stats.input_count = len(records)
        stats.input_chars = sum(len(r.get("text", "")) for r in records)

        logger.info("去重前: %d 条记录 / %d 字符", stats.input_count, stats.input_chars)

        # 阶段 1: 精确去重
        if self.config.enable_exact_dedup:
            records, stats.exact_removed = _exact_dedup(records)
            stats.after_exact = len(records)
            logger.info(
                "精确去重: 删除 %d 条 → 剩余 %d 条 (%.1f%%)",
                stats.exact_removed,
                stats.after_exact,
                100 * stats.after_exact / max(stats.input_count, 1),
            )
        else:
            stats.after_exact = stats.input_count

        # 阶段 2: 近似去重
        if self.config.enable_approx_dedup:
            records, stats.approx_removed = _approx_dedup(records, self.config)
            stats.after_approx = len(records)
            logger.info(
                "近似去重: 删除 %d 条 → 剩余 %d 条 (%.1f%%)",
                stats.approx_removed,
                stats.after_approx,
                100 * stats.after_approx / max(stats.after_exact, 1),
            )
        else:
            stats.after_approx = stats.after_exact

        # 写出
        stats.output_chars = sum(len(r.get("text", "")) for r in records)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_removed = stats.input_count - stats.after_approx
        logger.info(
            "去重完成: %d → %d 条 (去重率 %.1f%%) | 字符: %d → %d",
            stats.input_count,
            stats.after_approx,
            100 * total_removed / max(stats.input_count, 1),
            stats.input_chars,
            stats.output_chars,
        )
        return stats
