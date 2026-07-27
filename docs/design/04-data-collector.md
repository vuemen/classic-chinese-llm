# 数据采集器设计文档

**所属阶段:** Phase 2 — 数据管道
**涉及模块:** `src/classic_chinese_llm/data/collector.py` + `src/classic_chinese_llm/data/sources/`
**日期:** 2026-07-27

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | 多数据源适配 | 支持 5 个数据源（殆知阁、WikiSource、GitHub、四库全书、ctext.org），每个来源以统一接口接入 |
| F2 | 可插拔适配器 | 新增数据源只需新增一个适配器类，不修改 Collector 核心逻辑 |
| F3 | 统一采集流程 | 每个数据源执行 discover → parse → validate 三阶段流程 |
| F4 | 统一输出格式 | 所有来源输出统一 schema 的 JSONL，含 text + 元信息（source、title、era、genre 等） |
| F5 | 格式兼容 | 支持 txt、jsonl、xml 三种原始格式的解析 |
| F6 | 增量采集 | 支持断点续传——已完成的数据源跳过，不重复下载和解析 |
| F7 | 采集统计 | 采集完成后输出每个来源的文档数、字符数、成功率统计 |
| F8 | 数据目录组织 | 原始数据按 `data/raw/{source_name}/` 分目录存放，统一汇总到 `data/processed/collected.jsonl` |

### 1.2 非功能需求

- **性能**: 单线程采集即可（数据总量 ~3-6 亿字符，文本处理为主，非 CPU 密集型）
- **网络容错**: 下载失败自动重试 3 次，指数退避，不中断整个采集流程
- **编码兼容**: 自动检测文件编码（UTF-8/GBK/GB18030/Big5），避免乱码
- **可追溯**: 每行 JSONL 记录 `source` 和 `collected_at` 字段，可回溯到原始出处
- **不引入新依赖**: 使用项目已有的 `requests`、`beautifulsoup4`、`lxml`（data 可选依赖组）

---

## 2. 方案选型与对比

### 2.1 适配器模式

适配器模式的核心选择影响着整个采集器的扩展性。

| 方案 | 优势 | 劣势 | 结论 |
|------|------|------|------|
| **ABC 抽象基类** | IDE 补全友好、`isinstance` 检查、`@abstractmethod` 强制实现、mypy 类型检查 | 稍显冗长 | ✅ 选用 |
| typing.Protocol | 结构化类型、无需显式继承、更 Pythonic | 无法强制方法签名检查（运行时不做校验）、mypy 需额外配置 | ❌ |
| 注册表 + 函数 | 最灵活、无类继承约束 | 无类型安全、IDE 无法追踪、容易出错 | ❌ |

**最终选择**: **ABC 抽象基类**。理由：

```python
# ABC 方案 —— 类型安全，强制实现
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class SourceDocument:
    text: str
    source: str
    title: str
    metadata: dict  # {era, genre, author, url}

class BaseSource(ABC):
    @abstractmethod
    def discover(self) -> list[Path]: ...
    @abstractmethod
    def parse(self, path: Path) -> list[SourceDocument]: ...
    @abstractmethod
    def validate(self, doc: SourceDocument) -> bool: ...
```
对比 Protocol：
```python
# Protocol 方案 —— 缺少运行时强制
from typing import Protocol
class SourceLike(Protocol):
    def discover(self) -> list[Path]: ...
    # 如果子类忘记实现 parse()，运行时才会报错
```
项目已有 mypy strict 模式，ABC 的 `@abstractmethod` 在实例化时就会拦截遗漏的方法实现，是更安全的选择。

### 2.2 输出格式

| 格式 | 人类可读 | 流式处理 | 压缩比 | 生态支持 | 结论 |
|------|----------|----------|--------|----------|------|
| **JSONL** | ✅ 逐行可读 | ✅ 逐行追加 | 低 | 极广 | ✅ 选用 |
| Parquet | ❌ 需工具 | ❌ 整块读写 | 高 | datasets 库原生 | ❌ |
| CSV | 部分 | ✅ | 低 | 广 | ❌ 嵌套元信息难表示 |
| msgpack | ❌ 二进制 | ✅ | 中 | 一般 | ❌ |

**最终选择**: **JSONL**。每条记录一行 JSON，支持直接 `head`/`tail` 查看、逐行流式处理、追加写入无需重写全文件。对于 ~2-4 亿 token 的语料规模，JSONL 的存储开销约 2-3GB，完全可接受。

### 2.3 XML 解析（WikiSource 数据源专用）

| 方案 | 速度 | 内存 | API | 已有依赖 | 结论 |
|------|------|------|-----|----------|------|
| **lxml iterparse** | ⭐⭐⭐ | ⭐⭐⭐ 流式 | ⭐⭐ 稍复杂 | ✅ | ✅ 选用 |
| lxml etree | ⭐⭐ | ⭐ 全量加载 | ⭐⭐⭐ | ✅ | ❌ 大数据 OOM |
| ElementTree | ⭐ | ⭐ 全量加载 | ⭐⭐⭐ | 0（标准库） | ❌ 速度慢 |
| BeautifulSoup | ⭐ | ⭐ 全量加载 | ⭐⭐⭐ | ✅ | ❌ 不适合大 XML |

WikiSource XML dump 可达数百 MB，全量加载会导致内存溢出。

**最终选择**: **lxml iterparse**。流式解析，仅当前元素驻留内存。关键用法：

```python
from lxml import etree

for event, elem in etree.iterparse(xml_path, tag="{*}page"):
    title = elem.findtext("{*}title")
    text = elem.findtext(".//{*}text")
    if text:
        yield SourceDocument(text=text, source="wikisource", title=title, ...)
    elem.clear()  # 释放已处理元素的内存
```

### 2.4 文件编码检测

| 方案 | 准确率 | 速度 | 额外依赖 | 结论 |
|------|--------|------|----------|------|
| **chardet → 手动映射** | ⭐⭐⭐ | ⭐⭐ | 0（chardet 是 requests 子依赖） | ✅ 选用 |
| cchardet | ⭐⭐⭐ | ⭐⭐⭐ | cchardet | ❌ 需编译 |
| charset-normalizer | ⭐⭐⭐ | ⭐⭐ | charset-normalizer | ❌ 额外依赖 |
| 盲目尝试 GBK→UTF-8 | ⭐⭐ | ⭐⭐⭐ | 0 | ❌ 不可靠 |

**最终选择**: **chardet**。`requests` 已依赖 `chardet`，因此不增加新依赖。对于中文古籍常见编码（UTF-8、GBK、GB18030）检测准确率足够。

---

## 3. 最终方案

### 3.1 模块结构

```
src/classic_chinese_llm/data/
├── __init__.py                    # 导出 Collector, BaseSource, SourceDocument
├── collector.py                   # 采集编排器
├── schemas.py                     # SourceDocument + 输出 JSONL schema
├── sources/
│   ├── __init__.py                # 注册机制
│   ├── base.py                    # BaseSource 抽象基类
│   ├── daizhige.py                # 殆知阁 txt 适配器
│   ├── wikisource.py              # WikiSource XML 适配器
│   ├── github_corpora.py          # GitHub 开源语料适配器
│   ├── sikuquanshu.py             # 四库全书适配器
│   └── ctext.py                   # ctext.org 手动补充适配器
```

### 3.2 核心接口设计

#### 数据模型（schemas.py）

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

@dataclass
class SourceDocument:
    """统一文档模型 —— 所有数据源适配器的输出格式。"""

    text: str                                    # 正文内容
    source: str                                  # 来源标识: daizhige | wikisource | github | siku | ctext
    title: str = ""                              # 篇名/书名
    author: str = ""                             # 作者（朝代+人名，如 "唐·杜甫"）
    era: str = ""                                # 朝代: 先秦/两汉/魏晋南北朝/唐/宋/元/明/清
    genre: str = ""                              # 体裁: 经/史/子/集/诗/词/赋/散文/小说
    url: str = ""                                # 原始 URL
    chapter: str = ""                            # 章节/卷
    metadata: dict[str, Any] = field(default_factory=dict)  # 扩展元信息
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_jsonl_line(self) -> str:
        """序列化为一行 JSONL。"""
        import json
        return json.dumps(asdict(self), ensure_ascii=False)
```

#### 抽象基类（sources/base.py）

```python
from abc import ABC, abstractmethod
from pathlib import Path
from classic_chinese_llm.data.schemas import SourceDocument


class BaseSource(ABC):
    """数据源适配器基类。

    每个数据源实现三个核心方法：
    1. discover() — 发现可用的原始文件
    2. parse()    — 解析单个文件为 SourceDocument 列表
    3. validate() — 校验单个文档是否合格

    生命周期: discover → [parse → validate] × N
    """

    # 子类必须定义的标识
    name: str           # 来源名称，如 "daizhige"
    display_name: str   # 显示名称，如 "殆知阁"

    @abstractmethod
    def discover(self, raw_dir: Path) -> list[Path]:
        """在 raw_dir/{self.name}/ 中发现所有待处理的原始文件。

        返回值是按处理顺序排序的文件路径列表。
        空目录返回空列表（不报错），Collector 据此跳过该来源。
        """
        ...

    @abstractmethod
    def parse(self, file_path: Path) -> list[SourceDocument]:
        """解析一个原始文件，返回文档列表。

        Args:
            file_path: 单个原始文件的路径

        Returns:
            SourceDocument 列表。解析失败时返回空列表并 log warning。
        """
        ...

    def validate(self, doc: SourceDocument) -> bool:
        """校验单篇文档是否合格。默认: 正文长度 >= 10 字符。"""
        return len(doc.text.strip()) >= 10

    def post_process(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        """可选的采集后处理（去重、合并等），默认无操作。"""
        return docs
```

#### 采集编排器（collector.py）

```python
import json
import time
from pathlib import Path
from datetime import datetime

from classic_chinese_llm.data.sources.base import BaseSource
from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


class Collector:
    """数据采集编排器 —— 遍历所有数据源，输出统一 JSONL。"""

    def __init__(
        self,
        sources: list[BaseSource],
        raw_dir: str | Path,
        output_dir: str | Path,
        *,
        retry_attempts: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.sources = sources
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

        self._stats: dict[str, dict] = {}  # {source_name: {files, docs, chars}}

    def run(self) -> Path:
        """执行全量采集，返回汇总 JSONL 路径。"""
        output_path = self.output_dir / "collected.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        total_docs = 0
        with open(output_path, "w", encoding="utf-8") as out:
            for source in self.sources:
                logger.info("处理数据源: %s", source.display_name)
                docs = self._collect_source(source)
                for doc in docs:
                    out.write(doc.to_jsonl_line() + "\n")
                total_docs += len(docs)

        self._print_stats()
        logger.info("采集完成 | 总计 %d 篇文档 → %s", total_docs, output_path)
        return output_path

    def _collect_source(self, source: BaseSource) -> list[SourceDocument]:
        """采集单个数据源（含重试逻辑）。"""
        src_dir = self.raw_dir / source.name
        src_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: 发现文件
        files = source.discover(src_dir)
        if not files:
            logger.warning("  %s: 无文件发现，跳过", source.display_name)
            self._stats[source.name] = {"files": 0, "docs": 0, "chars": 0, "valid": 0}
            return []

        logger.info("  %s: 发现 %d 个文件", source.display_name, len(files))

        # Phase 2 & 3: 解析 + 校验（每个文件带重试）
        all_docs: list[SourceDocument] = []
        for fp in files:
            for attempt in range(1, self.retry_attempts + 1):
                try:
                    parsed = source.parse(fp)
                    valid = [d for d in parsed if source.validate(d)]
                    all_docs.extend(valid)
                    break  # 成功，跳出重试
                except Exception:
                    if attempt == self.retry_attempts:
                        logger.exception(
                            "  %s: 解析失败（重试 %d 次后跳过）: %s",
                            source.display_name, self.retry_attempts, fp.name
                        )
                    else:
                        wait = self.retry_backoff ** attempt
                        logger.warning("  %s: 重试 %d/%d（%.0fs 后退避）: %s",
                                       source.display_name, attempt, self.retry_attempts, wait, fp.name)
                        time.sleep(wait)

        # 后处理
        all_docs = source.post_process(all_docs)

        total_chars = sum(len(d.text) for d in all_docs)
        self._stats[source.name] = {
            "files": len(files), "docs": len(all_docs), "chars": total_chars,
            "valid": len(all_docs),
        }
        logger.info("  %s: 完成 → %d 篇 / %d 字符", source.display_name, len(all_docs), total_chars)
        return all_docs

    def _print_stats(self) -> None:
        """打印采集统计报告。"""
        logger.info("=" * 50)
        logger.info("采集统计报告")
        logger.info("=" * 50)
        for name, stat in self._stats.items():
            chars_m = stat["chars"] / 1_000_000
            logger.info("  %15s: %4d 文件 → %6d 篇 / %8.1fM 字符", name, stat["files"], stat["docs"], chars_m)
```

### 3.3 单个数据源适配器示例

```python
# sources/daizhige.py

import re
import chardet
from pathlib import Path

from classic_chinese_llm.data.sources.base import BaseSource
from classic_chinese_llm.data.schemas import SourceDocument
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


class DaiZhiGeSource(BaseSource):
    """殆知阁古代汉语语料库适配器。

    原始格式: 打包的 txt 文件，每行一篇文档。
    常见结构: 【篇名】正文内容
    """

    name = "daizhige"
    display_name = "殆知阁"

    # 篇名标记正则（适配多种格式）
    TITLE_PATTERN = re.compile(
        r"^[【\[《](.+?)[】\]》]\s*"  # 【篇名】或 [篇名]
        r"|"
        r"^(.+?)\s*\n",             # 首行作为标题（无标记时）
        re.MULTILINE,
    )

    def discover(self, raw_dir: Path) -> list[Path]:
        """发现所有 txt 文件，按文件名排序。"""
        files = sorted(raw_dir.glob("**/*.txt"))
        # 排除 README、说明文件
        files = [f for f in files if "readme" not in f.name.lower() and "说明" not in f.name]
        return files

    def parse(self, file_path: Path) -> list[SourceDocument]:
        """解析 txt 文件：自动检测编码，按篇目拆分。"""
        raw_bytes = file_path.read_bytes()
        encoding = chardet.detect(raw_bytes)["encoding"] or "utf-8"
        text = raw_bytes.decode(encoding, errors="replace")

        docs = []
        # 按双换行切分文档（古文段落间通常以空行分隔）
        paragraphs = re.split(r"\n\s*\n", text)
        for para in paragraphs:
            para = para.strip()
            if len(para) < 10:
                continue

            # 尝试提取标题
            title = ""
            match = self.TITLE_PATTERN.match(para)
            if match:
                title = match.group(1) or match.group(2) or ""
                body = para[match.end():].strip()
            else:
                body = para

            if body:
                docs.append(SourceDocument(
                    text=body,
                    source=self.name,
                    title=title.strip(),
                    metadata={"file": file_path.name, "encoding": encoding},
                ))

        return docs
```

### 3.4 脚本入口（scripts/collect_data.py 骨架）

```python
"""数据采集 CLI 入口。"""

import argparse
from pathlib import Path

from classic_chinese_llm.config.paths import PathConfig
from classic_chinese_llm.utils.logging_config import setup_logging
from classic_chinese_llm.data.collector import Collector
from classic_chinese_llm.data.sources.daizhige import DaiZhiGeSource
from classic_chinese_llm.data.sources.wikisource import WikiSourceSource
# ... 其他 source 导入


def main():
    parser = argparse.ArgumentParser(description="Classical Chinese LLM — 数据采集")
    parser.add_argument("--raw-dir", default="data/raw", help="原始数据根目录")
    parser.add_argument("--output-dir", default="data/processed", help="汇总 JSONL 输出目录")
    parser.add_argument("--sources", nargs="*", help="指定采集的数据源（默认全部）")
    parser.add_argument("--retry", type=int, default=3, help="失败重试次数")
    args = parser.parse_args()

    PathConfig.initialize(project_root=".")
    setup_logging(level="INFO")

    all_sources = [
        DaiZhiGeSource(),
        WikiSourceSource(),
        # GitHubSource(),
        # SiKuSource(),
        # CtextSource(),
    ]

    # 按需过滤
    if args.sources:
        all_sources = [s for s in all_sources if s.name in args.sources]

    collector = Collector(
        sources=all_sources,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        retry_attempts=args.retry,
    )
    collector.run()


if __name__ == "__main__":
    main()
```

---

## 4. 关键技术点

### 4.1 三阶段采集流程（discover → parse → validate）

这一设计将数据源的三个关注点分离：

- **discover**: "有哪些文件要处理？" — 关注文件系统操作、glob 匹配、排除规则
- **parse**: "文件里有什么内容？" — 关注格式解析、编码处理、文本提取
- **validate**: "这份文档合格吗？" — 关注质量过滤、长度检查、完整性校验

分离后，每个方法单一职责，便于单独测试和覆盖。例如 `validate` 可以独立对所有 source 进行单元测试。

### 4.2 编码自动检测

中文古籍最常见的三种编码及检测特征：

| 编码 | 特征 | 覆盖语料 |
|------|------|----------|
| UTF-8 | BOM `\xEF\xBB\xBF` 或无 | 现代整理的语料 |
| GBK | 双字节，高字节 `\x81-\xFE` | 国内早期数字化文本 |
| GB18030 | 兼容 GBK，四字节拓展 | 国家标准的古籍数字化 |

`chardet.detect()` 先读取文件前 10KB 采样，以其推测编码。当 `chardet` 不确定时（confidence < 0.5），fallback 顺序为 UTF-8 → GB18030 → GBK。

### 4.3 增量采集与幂等性

Collector 本身不实现增量采集——它依赖上游数据源适配器的 `discover()` 方法过滤已处理文件：

```python
def discover(self, raw_dir: Path) -> list[Path]:
    all_files = sorted(raw_dir.glob("**/*.txt"))
    done_file = raw_dir / ".collected_files.txt"
    if done_file.exists():
        done = set(done_file.read_text().splitlines())
        return [f for f in all_files if f.name not in done]
    return all_files
```

这样 Collector 保持简单，增量逻辑由各适配器按需实现。

### 4.4 WikiSource XML 流式解析

WikiSource 的 XML dump 格式为：

```xml
<mediawiki>
  <page>
    <title>论语</title>
    <revision>
      <text xml:space="preserve">子曰：学而时习之...</text>
    </revision>
  </page>
  <page>...</page>
</mediawiki>
```

关键挑战是 namespace 处理。WikiSource dump 可能带或不带 namespace 前缀。`lxml.iterparse` 的 `tag="{*}page"` 通配符可同时匹配两种。

`elem.clear()` 是关键的内存优化——处理完一个 `<page>` 后立即释放其子元素树，保证数百 MB 的 XML 也能在数 GB 内存内完成解析。

### 4.5 数据源注册机制

当前 5 个数据源通过硬编码列表注册。随着来源增多，可引入基于 `importlib` 的自动发现：

```python
# sources/__init__.py
from classic_chinese_llm.data.sources.base import BaseSource

_registry: dict[str, type[BaseSource]] = {}

def register_source(cls: type[BaseSource]) -> type[BaseSource]:
    _registry[cls.name] = cls
    return cls

def get_source(name: str) -> BaseSource:
    return _registry[name]()
```

但 Phase 2 中 5 个来源手写即可，自动注册留待来源数 >10 时再引入。

---

## 5. 与其他模块的关系

```
Config ─── 被依赖 ───> Collector (PathConfig 提供 raw_dir / output_dir)
Utils  ─── 被依赖 ───> Collector (logging)
Collector ─── 输出 ───> Cleaner (data/processed/collected.jsonl)
```

Collector 是 Phase 2 数据管道的起点。它的输出文件 `collected.jsonl` 是 Cleaner 的输入。当前 `PathConfig` 已有 `raw_data_dir` 和 `processed_data_dir` 属性，Collector 直接使用。

---

## 6. 验证清单

- [ ] `BaseSource` 的 `@abstractmethod` 正确拦截未实现的方法
- [ ] 殆知阁 txt 适配器正确检测 GBK/UTF-8 编码并解析
- [ ] WikiSource XML 适配器流式解析 200MB+ dump 不 OOM
- [ ] 所有适配器的输出 `SourceDocument.source` 字段正确
- [ ] Collector 在有 0 个文件的来源时不崩溃（skip）
- [ ] 解析失败重试 3 次后跳过该文件，继续处理下一个
- [ ] 最终输出 JSONL 每行可被 `json.loads` 正确解析
- [ ] `_print_stats()` 输出的统计数字与实际文件数一致
