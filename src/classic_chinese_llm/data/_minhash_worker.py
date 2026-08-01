"""MinHash 并行构建的独立 Worker 模块。

此模块刻意不导入 classic_chinese_llm 的任何模块，
避免 Windows spawn 模式下每个子进程加载 torch 等重型依赖。
仅依赖 datasketch + 标准库。
"""

from __future__ import annotations

import re
from typing import Any

from datasketch import MinHash

# 空白字符压缩（与 deduplicator._char_shingles 保持逻辑一致）
_RE_WHITESPACE = re.compile(r"\s+")


def _char_shingles(text: str, k: int = 5) -> set[str]:
    """生成字符级 k-gram shingle 集合。"""
    clean = _RE_WHITESPACE.sub(" ", text)
    if len(clean) < k:
        return set()
    return {clean[i : i + k] for i in range(len(clean) - k + 1)}


def build_single_minhash(
    args: tuple[int, str, int, int, int],
) -> tuple[int, bytes]:
    """构建单个文档的 MinHash 签名。

    Args:
        args: (index, text, num_perm, seed, shingle_size)

    Returns:
        (index, hashvalues_digest_bytes) — 主进程中通过
        MinHash(num_perm, seed, hashvalues=digest, scheme=scheme) 重建
    """
    idx, text, num_perm, seed, shingle_size = args
    m = MinHash(num_perm=num_perm, seed=seed)
    shingles = _char_shingles(text, k=shingle_size)
    if not shingles:
        m.update(b"")
    else:
        for s in shingles:
            m.update(s.encode("utf-8"))
    return idx, m.digest()


__all__ = ["build_single_minhash"]
