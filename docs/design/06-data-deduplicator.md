# 数据去重器设计文档

**所属阶段:** Phase 2 — 数据管道
**涉及模块:** `src/classic_chinese_llm/data/deduplicator.py`
**日期:** 2026-07-27

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | 精确去重 | 基于 SHA-256 哈希值识别完全相同的文档（byte-level duplicate） |
| F2 | 近似去重 | 基于 MinHash + LSH 识别高度相似的文档对（Jaccard 相似度 ≥0.85） |
| F3 | 可配置阈值 | Jaccard 阈值、MinHash permutation 数量、LSH band/row 参数均可配置 |
| F4 | 去重策略选择 | 在近似重复文档组中，支持"保留最长"、"保留最早来源"、"保留最完整元信息"等策略 |
| F5 | 去重统计 | 输出精确去重数、近似去重数、总保留数、去重率 |
| F6 | 内存友好 | 支持流式处理，对约 7-9 亿 token 规模语料不会 OOM（12GB RAM 场景） |
| F7 | 确定性 | 相同输入 + 相同 seed → 相同输出（MinHash 使用固定 seed） |

### 1.2 非功能需求

- **性能**: 两阶段去重在数分钟内完成（精确去重 ~秒级，近似去重 ~分钟级）
- **可复现**: 固定 seed 保证每次运行结果一致
- **增量友好**: 设计上允许未来扩展为增量去重（新文档与已有文档比较）
- **最小依赖**: 使用 `datasketch`（data 可选依赖组已声明），不引入额外库

---

## 2. 方案选型与对比

### 2.1 近似去重算法

这是去重器最核心的技术决策。

| 方案 | 精度 | 速度 | 内存 | 理论基础 | 结论 |
|------|------|------|------|----------|------|
| **MinHash + LSH** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ 可控 | Jaccard 无偏估计 | ✅ 选用 |
| SimHash | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Cosine/Hamming | ❌ 中文效果差 |
| 直接 n-gram 比较 | ⭐⭐⭐ | ⭐ | ⭐ O(n²) | 精确 | ❌ 不可行 |
| 向量嵌入 + ANN | ⭐⭐ | ⭐ | ⭐ | 语义相似 | ❌ 过度设计 |

**详细对比**:

**MinHash + LSH（选用）**:
```python
from datasketch import MinHash, MinHashLSH

# 将文档文本转为 MinHash 签名
m = MinHash(num_perm=128)
for shingle in shingles(text):
    m.update(shingle.encode("utf-8"))

# 将签名插入 LSH 索引
lsh = MinHashLSH(threshold=0.85, num_perm=128)
lsh.insert(doc_id, m)

# 查询相似文档
result = lsh.query(m)  # → [doc_id_1, doc_id_2, ...]
```
✅ 优势: MinHash 对 Jaccard 相似度的估计是**无偏的**；LSH 将 O(n²) 的 pairwise 比较降为 O(n)；参数（num_perm）可调节精度/速度的权衡

**SimHash**:
```python
# 产生固定长度的 bit 指纹，通过 Hamming 距离判断相似
# 问题: 对中文短文本（<100字）区分能力弱，Hamming 距离与 Jaccard 无直接关系
```
❌ 劣势: SimHash 擅长检测"几乎完全相同"的文档（Hamming distance ≤3），但文言文去重的核心场景是"同一段落的不同版本"（如不同来源引用的同一段《论语》），相似度在 0.85-0.95 区间，MinHash 对此区间的分辨能力更好。

**直接 n-gram 比较**: O(n²) 复杂度在 100K+ 文档时即不可行，无需讨论。

**向量嵌入 + ANN (Approximate Nearest Neighbor)**: 将文档转为语义 embedding → 使用 FAISS 做 ANN 搜索。适用于"语义相似但用词不同"的场景，不适合本项目的需求（检测字符级文本复用）。

**最终选择: MinHash + LSH**（datasketch 库实现）。

### 2.2 Shingle (n-gram) 粒度

MinHash 依赖将文本分解为 shingle（重叠的 n-gram），shingle 粒度的选择直接影响去重效果。

| 粒度 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **字符 5-gram** | 对中文自然（5 字≈1 个短语）、抗词序变化好 | shingle 数量多，计算量稍大 | ✅ 选用 |
| 字符 3-gram | 更快、更敏感（易匹配短片段） | 误匹配率高——3 个汉字可出现在完全不同的语境中 | ❌ |
| 词级 n-gram | 语义准确、有分词信息 | 依赖分词器、文言文分词困难、未登录词多 | ❌ |
| 字符 8-gram | 精确、抗噪声 | 对插入/删除敏感，丢失较短的匹配 | ❌ 备选 |

**最终选择: 字符 5-gram**。文言文没有标准的分词方案（不同算法结果差异大），字符级 shingle 避免了分词依赖。5 个汉字在文言文中约等于一个短语（如 `学而时习之`、`温故而知新`），是区分文本相似度的合适粒度。

```python
def char_shingles(text: str, k: int = 5) -> set[str]:
    """生成字符级 k-gram shingle 集合。"""
    clean = text.replace("\n", "").replace(" ", "")
    return {clean[i:i+k] for i in range(len(clean) - k + 1)}
```

### 2.3 MinHash 实现库

| 方案 | 速度 | 维护状态 | 集成难度 | 结论 |
|------|------|----------|----------|------|
| **datasketch** | C 优化 | ✅ 活跃 | 低（pip install，纯 Python + C） | ✅ 选用 |
| 自实现 MinHash | 中等 | — | 中（需自测正确性） | ❌ 重复造轮子 |
| scipy + numpy | 较快 | ✅ | 中（需自行实现 LSH） | ❌ 工作量大 |

**最终选择: datasketch**。它是学术界和工业界广泛使用的 MinHash 库，Google 引用超过 2000 次。项目 `pyproject.toml` 的 `data` 可选依赖组已声明 `datasketch>=1.6`。

---

## 3. 最终方案

### 3.1 模块结构

```
src/classic_chinese_llm/data/
├── __init__.py
├── collector.py
├── cleaner.py
├── deduplicator.py     # Deduplicator 两阶段去重编排器
└── ...
```

Deduplicator 为单文件模块，包含 `DeduplicatorConfig`、精确去重逻辑、MinHash+LSH 近似去重逻辑。

### 3.2 核心接口设计

```python
# data/deduplicator.py

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from datasketch import MinHash, MinHashLSH

from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


# ─── 配置 ──────────────────────────────────────────────────────────────


@dataclass
class DeduplicatorConfig:
    """去重器参数配置。"""

    # MinHash 参数
    num_perm: int = 128              # permutation 数量（越大越精确）
    shingle_size: int = 5            # 字符 n-gram 大小
    jaccard_threshold: float = 0.85  # Jaccard 相似度阈值

    # MinHash seed（固定保证可复现）
    seed: int = 42

    # LSH 参数（由 jaccard_threshold 和 num_perm 自动推导）
    # 默认不显式设置 bands/rows，使用 datasketch 内置优化
    lsh_num_bands: int | None = None
    lsh_num_rows: int | None = None

    # 近似去重组内的策略
    keep_strategy: str = "longest"   # "longest" | "earliest" | "most_complete"

    # 是否启用各阶段
    enable_exact_dedup: bool = True
    enable_approx_dedup: bool = True


# ─── Shingle 生成 ──────────────────────────────────────────────────────


def _char_shingles(text: str, k: int = 5) -> set[str]:
    """生成字符级 k-gram shingle 集合。

    预处理：去除空白字符，保留标点符号。
    标点在文言文中有实际语义价值（句读），不应移除。
    """
    clean = text.replace("\n", " ").replace("\r", " ")
    # 移除连续空格
    while "  " in clean:
        clean = clean.replace("  ", " ")
    return {clean[i:i + k] for i in range(len(clean) - k + 1)}


# ─── 精确去重 ──────────────────────────────────────────────────────────


def _compute_sha256(text: str) -> str:
    """计算文本的 SHA-256 哈希值。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _exact_dedup(
    records: list[dict],
) -> tuple[list[dict], int]:
    """精确去重：相同 SHA-256 → 保留第一条。

    Returns:
        (deduped_records, removed_count)
    """
    seen: set[str] = set()
    deduped: list[dict] = []
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


def _build_minhash(text: str, num_perm: int, seed: int, shingle_size: int) -> MinHash:
    """为单个文档构建 MinHash 签名。"""
    m = MinHash(num_perm=num_perm, seed=seed)
    shingles = _char_shingles(text, k=shingle_size)
    if not shingles:
        # 极短文档：编码空字符串标记
        m.update(b"")
    else:
        for s in shingles:
            m.update(s.encode("utf-8"))
    return m


def _approx_dedup(
    records: list[dict],
    config: DeduplicatorConfig,
) -> tuple[list[dict], int]:
    """近似去重：MinHash + LSH 检测相似文档对。

    流程：
    1. 为每条文档构建 MinHash 签名
    2. 将所有签名插入 LSH 索引
    3. 对每条文档查询 LSH 获取候选相似文档
    4. 对候选对计算实际 Jaccard 相似度，超过阈值的归为一组
    5. 每组按 keep_strategy 保留一条

    Returns:
        (deduped_records, removed_count)
    """
    num_perm = config.num_perm
    threshold = config.jaccard_threshold

    # 构建 LSH 索引
    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm,
        params=(config.lsh_num_bands, config.lsh_num_rows),
    )

    # 构建 MinHash 签名
    minhashes: dict[int, MinHash] = {}
    for idx, record in enumerate(records):
        m = _build_minhash(
            record.get("text", ""),
            num_perm=num_perm,
            seed=config.seed,
            shingle_size=config.shingle_size,
        )
        minhashes[idx] = m
        lsh.insert(idx, m)

    # 查询 LSH 获取候选相似文档，构建连通图
    from collections import defaultdict

    graph: dict[int, set[int]] = defaultdict(set)
    for idx in range(len(records)):
        candidates = lsh.query(minhashes[idx])
        for cand in candidates:
            if cand != idx:
                # 计算精确 Jaccard（避免 LSH 假阳性）
                actual_jaccard = minhashes[idx].jaccard(minhashes[cand])
                if actual_jaccard >= threshold:
                    graph[idx].add(cand)
                    graph[cand].add(idx)

    # 通过 DFS/Union-Find 将连通分量分组
    visited: set[int] = set()
    groups: list[list[int]] = []

    for idx in range(len(records)):
        if idx in visited:
            continue
        # BFS 收集连通分量
        group: list[int] = []
        stack = [idx]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            group.append(node)
            stack.extend(graph[node] - visited)
        groups.append(group)

    # 每组保留一条
    deduped: list[dict] = []
    removed = 0
    for group in groups:
        if len(group) == 1:
            deduped.append(records[group[0]])
        else:
            keeper = _select_keeper(group, records, config.keep_strategy)
            deduped.append(records[keeper])
            removed += len(group) - 1

    return deduped, removed


def _select_keeper(
    group: list[int],
    records: list[dict],
    strategy: str,
) -> int:
    """从重复文档组中选择保留哪一条。

    策略:
    - "longest": 保留正文最长的（信息量最大）
    - "earliest": 保留在原始列表中位置最靠前的
    - "most_complete": 保留元信息字段最多的
    """
    if strategy == "earliest":
        return min(group)
    elif strategy == "most_complete":
        return max(group, key=lambda i: sum(
            1 for v in records[i].values() if v
        ))
    else:  # "longest" (default)
        return max(group, key=lambda i: len(records[i].get("text", "")))


# ─── 去重编排器 ────────────────────────────────────────────────────────


@dataclass
class DedupStats:
    """去重统计信息。"""

    input_count: int = 0
    after_exact: int = 0        # 精确去重后剩余
    exact_removed: int = 0
    after_approx: int = 0       # 近似去重后剩余
    approx_removed: int = 0
    input_chars: int = 0
    output_chars: int = 0


class Deduplicator:
    """两阶段去重编排器。

    阶段 1: SHA-256 精确去重（字节级完全相同 → 删除）
    阶段 2: MinHash + LSH 近似去重（Jaccard ≥ threshold → 保留一条）

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

        # 加载全部记录到内存（JSONL 规模 ~100K-500K 条，每条平均 ~500 字符）
        # 总计约 50-250MB，12GB RAM 完全充裕
        with open(input_path, encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

        stats.input_count = len(records)
        stats.input_chars = sum(len(r.get("text", "")) for r in records)

        logger.info("去重前: %d 条记录 / %d 字符", stats.input_count, stats.input_chars)

        # ── 阶段 1: 精确去重 ──
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

        # ── 阶段 2: 近似去重 ──
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

        # 写出去重后数据
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
```

### 3.3 LSH 参数自动推导

datasketch 的 `MinHashLSH` 在给定 `threshold` 和 `num_perm` 时可自动选择最优的 `(bands, rows)` 配置。下表展示不同 `num_perm` 值对应的推荐配置：

| num_perm | bands (b) | rows (r) | b × r | S-curve 陡峭度 | 适用场景 |
|----------|-----------|----------|-------|----------------|----------|
| 64 | 8 | 8 | 64 | 较平缓 | 快速粗略扫描 |
| **128** | **16** | **8** | **128** | **适中** | **推荐（平衡精度/速度）** |
| 256 | 32 | 8 | 256 | 更陡峭 | 高精度需求 |

选择 num_perm=128 的理由：
- S-curve 在 threshold=0.85 处足够陡峭（假阳性率低）
- 每个 MinHash 签名仅 ~1KB（128 × 8 bytes），200K 文档约 200MB
- datasketch LSH 查询在 200K 级规模下 <10 秒

### 3.4 使用示例

```python
from classic_chinese_llm.data.deduplicator import Deduplicator, DeduplicatorConfig

config = DeduplicatorConfig(
    num_perm=128,
    shingle_size=5,
    jaccard_threshold=0.85,
    keep_strategy="longest",
    enable_exact_dedup=True,
    enable_approx_dedup=True,
)

dedup = Deduplicator(config)
stats = dedup.deduplicate(
    "data/processed/cleaned.jsonl",
    "data/processed/deduplicated.jsonl",
)
print(f"去重率: {(stats.input_count - stats.after_approx) / stats.input_count:.1%}")
```

---

## 4. 关键技术点

### 4.1 两阶段去重的计算经济学

为什么要先精确去重再做近似去重？

```
精确去重:  O(n) 时间, O(unique_count) 内存, ~秒级
近似去重:  O(n) 时间, O(n × num_perm × 8B) 内存, ~分钟级

先精确去重: 减少 n → 减少近似去重阶段的计算量和内存
```

以 200K 文档为例：
- 精确去重通常可减少 10-20% 文档（不同来源之间的完全拷贝）
- 剩余 ~160K-180K 文档进入近似去重，节省 ~10-20% 计算成本

且精确去重帮助减少近似去重中的"连通分量膨胀"——如果同一个文本在 5 个来源中完全相同，精确去重只留 1 条，避免这 5 条在近似去重阶段形成 5-clique。

### 4.2 字符级 Shingle 对中文的特殊意义

中文文本与英文在 n-gram 去重上的关键差异：

| 属性 | 英文 | 中文（文言文） |
|------|------|----------------|
| 词边界 | 空格分隔，明确的 token | 无空格，需分词（困难） |
| 字符信息量 | 1 字符 ≈ 1 字母（低信息量） | 1 字符 ≈ 1 语素（高信息量） |
| n-gram 选择 | 通常用 word 3-gram 或 char 5-gram | 字符 5-gram ≈ 英文 word 3-gram |
| Shingle 冲突率 | 低（大字母表 + 低信息密度） | 极低（大字符集 + 高信息密度） |

文言文的 CJK 统一汉字约 20,000+ 个常用字，5-gram 的理论空间为 20,000⁵——shingle 冲突几乎不可能发生。这使得字符级 MinHash 在中文场景中特别有效。

### 4.3 num_perm 的精度-内存权衡

MinHash 估计 Jaccard 相似度的标准误差：

```
SE ≈ sqrt(J × (1 - J) / num_perm)
```

对于 J=0.85（我们的阈值），不同 num_perm 的 95% 置信区间：

| num_perm | SE | 95% CI for J=0.85 |
|----------|-----|-------------------|
| 64 | 0.045 | [0.762, 0.938] |
| **128** | **0.032** | **[0.788, 0.912]** |
| 256 | 0.022 | [0.806, 0.894] |

128 个 permutation 在 J=0.85 处的 95% CI 为 ±0.062，足够分辨"高度相似"（≥0.85）和"不那么相似"（<0.80）的文档。如需更高精度，可增加至 256，代价是签名大小翻倍。

### 4.4 短文本的去重处理

极短文档（<30 字符）存在两个问题：

1. Shingle 数量少：20 字的文本只有 16 个 5-gram，MinHash 估计方差大
2. Jaccard 分母小：两个 20 字文本共享 10 个字时 Jaccard = 0.5，但实际可能是一段正文的碎片

策略：
- 对 <30 字符的文档，跳过近似去重阶段（仅参与精确去重）
- LSH 构建时过滤掉 shingle 数量 <10 的文档，避免它们产生大量噪音候选
- 统计报告中标记"因过短跳过近似去重"的文档数

### 4.5 keep_strategy 的语义

当一组文档被判定为近似重复时，选择保留哪一条影响最终数据质量：

| 策略 | 语义 | 适用场景 |
|------|------|----------|
| `longest` | 保留正文最长的 | 默认策略——信息量最大 |
| `earliest` | 保留最先出现的 | 偏好某个数据源的排序 |
| `most_complete` | 保留元信息最完整的 | 后续 SFT 需要 title/era/author 等元信息 |

推荐默认使用 `longest`，因为较长的文本版本通常是较完整的版本（注释、整理后的版本往往比纯原文更长且有附加信息）。

### 4.6 内存估算与 12GB 约束

200K 条文档的去重内存预算：

```
原始 JSONL 加载:      200K × ~800B ≈ 160 MB
MinHash 签名 (128 perm): 200K × 128 × 8B ≈ 205 MB
LSH 索引 overhead:     ~50 MB (hashtable 开销)
─────────────────────────────────────────
总计:                  ~415 MB
```

在 12GB RAM 场景中仅占 ~3.5%，充裕度足够未来语料规模增长 10 倍。

---

## 5. 与其他模块的关系

```
Config ─── 被依赖 ───> Deduplicator (DeduplicatorConfig 注入)
Utils  ─── 被依赖 ───> Deduplicator (logging)

Cleaner ─── 输出 cleaned.jsonl ──→ Deduplicator ──→ 输出 deduplicated.jsonl ──→ Formatter
                                                                           ──→ Pretrain (Phase 4)
```

Deduplicator 的输出有两条下游路径：
- → **Formatter**（指令数据集构建，Phase 5 SFT 用）
- → **Pretrain**（预训练数据集，Phase 4 直接使用）

去重后的 JSONL 本质上是"干净的文言文原文集合"，既可以交给 Formatter 构建指令数据，也可以直接作为预训练语料。

---

## 6. 验证清单

- [ ] `_char_shingles("子曰学而时习之", k=5)` 返回正确的 5-gram 集合
- [ ] 两条完全相同的文本 `_compute_sha256` 结果一致，不同文本结果不同
- [ ] `_exact_dedup` 输入 3 条相同 + 2 条不同的 5 条记录，返回 3 条
- [ ] 两条仅差 1 个字的文本 Jaccard ≥0.95，被近似去重归为一组
- [ ] 两条完全不同主题的文本 Jaccard <0.3，不被归为一组
- [ ] `seed=42` 固定时，同一数据集的近似去重每次结果相同
- [ ] `keep_strategy="longest"` 时，重复组中保留的确实是 text 最长的那条
- [ ] 输入 JSONL 和输出 JSONL 的每行都可通过 `json.loads` 解析
- [ ] 空输入时（0 条记录）不报错，统计数字全为 0
- [ ] `DedupStats` 的 `input_count == after_approx + exact_removed + approx_removed`
