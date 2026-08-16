# 数据格式化器设计文档

**所属阶段:** Phase 2 — 数据管道
**涉及模块:** `src/classic_chinese_llm/data/formatter.py`
**日期:** 2026-07-27

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | 指令-响应对生成 | 将去重后的文言文段落通过模板转换为 `(instruction, response)` 对 |
| F2 | 多任务类型 | 支持 7+ 种任务类型：文言→白话翻译、白话→文言翻译、文本续写、主旨概括、词句释义、语法分析、诗词创作 |
| F3 | ChatML 格式输出 | 输出标准 ChatML 格式 JSONL，每行含 `messages` 列表（system + user + assistant）|
| F4 | 系统提示词 | 为每种任务类型设计专用的 system prompt，塑造文言文专家 persona |
| F5 | 任务均衡采样 | 各任务类型按权重采样，避免单一类型占比过高 |
| F6 | 质量过滤 | 过滤生成的指令-响应对中明显不合格的样本（响应过短、模板未填充等）|
| F7 | Train/Val 切分 | 按比例切分训练集和验证集，输出到独立文件 |
| F8 | 数据统计 | 输出各任务类型的样本数、平均长度、占比等统计信息 |

### 1.2 非功能需求

- **确定性**: 相同 seed + 相同输入 → 相同输出（shuffle 和 split 使用固定 seed）
- **可扩展**: 新增任务类型只需新增模板函数，不修改 Formatter 核心逻辑
- **与 SFT 对接**: 输出格式直接兼容 HF `tokenizer.apply_chat_template`，下游 SFT 零转换成本
- **零外部依赖**: 纯 Python 字符串模板，不引入 Jinja2 等模板引擎

---

## 2. 方案选型与对比

### 2.1 对话格式

ChatML 代表 OpenAI/HuggingFace 的通用对话格式，是最广泛支持的标准。

| 格式 | 结构 | Tokenizer 兼容 | 生态支持 | 结论 |
|------|------|----------------|----------|------|
| **ChatML** | `{"messages": [{"role":..., "content":...}]}` | ✅ `apply_chat_template` | HF 原生、OpenAI 兼容 | ✅ 选用 |
| Alpaca | `{"instruction":..., "input":..., "output":...}` | ⚠️ 需手动转换 | LLaMA-Factory 等 | ❌ 需转换 |
| ShareGPT | `{"conversations": [{"from":..., "value":...}]}` | ⚠️ 需手动转换 | Vicuna 生态 | ❌ 过时 |

**最终选择: ChatML**。

```json
// ChatML 格式（选用）
{
  "messages": [
    {"role": "system", "content": "你是一位精通中国古代文学的文言文专家..."},
    {"role": "user", "content": "请将以下文言文翻译成白话文：\n\n子曰：学而时习之，不亦说乎？"},
    {"role": "assistant", "content": "孔子说：学习了知识后经常温习实践，不也是很快乐的事吗？"}
  ]
}
```

ChatML 的核心优势：HF `tokenizer.apply_chat_template()` 可直接将其序列化为模型输入 token 序列，包含 `system`/`user`/`assistant` 的分隔 token。下游 SFT 训练（Phase 5）无需任何格式转换。

### 2.2 模板引擎

| 方案 | 学习成本 | 类型安全 | 额外依赖 | 维护性 | 结论 |
|------|----------|----------|----------|--------|------|
| **Python str.format / f-string** | 0 | mypy 检查 | 0 | ⭐⭐⭐ | ✅ 选用 |
| Jinja2 | 中 | ❌ 运行时 | jinja2 | ⭐⭐ | ❌ 过重 |
| 手写拼接 | 0 | ❌ | 0 | ⭐ | ❌ 可读性差 |
| YAML 模板 | 低 | ❌ | 0（已有 pyyaml） | ⭐⭐ | 备选 |

**最终选择: Python str.format + 模板函数**。文言文指令模板通常不超过 3-5 行，str.format 完全满足。不需要 Jinja2 的条件/循环等高级特性。主要优势是 mypy 可以检查模板参数的类型正确性。

### 2.3 指令数据生成策略

这是决定 SFT 数据质量上限的关键决策。

| 方案 | 质量 | 多样性 | 成本 | 可行性 | 结论 |
|------|------|--------|------|--------|------|
| **静态模板 + 规则组合** | ⭐⭐ 中等 | ⭐⭐ 中等 | 零 | ✅ Phase 2 可行 | ✅ 选用 |
| LLM Self-Instruct | ⭐⭐⭐ | ⭐⭐⭐ | API 费用 | ❌ 无可用 LLM | ❌ 本阶段不可行 |
| 人工标注 | ⭐⭐⭐ 最高 | ⭐ 低 | 极高 | ❌ 资源不足 | ❌ |
| 现有数据集转换 | ⭐⭐ | ⭐⭐ | 低 | ⚠️ 文言文 SFT 数据集极少 | 备选 |

**最终选择: 静态模板 + 规则组合**。理由：

1. **现实约束**: Phase 2 阶段还没有可用的文言文 LLM，无法做 Self-Instruct；文言文 SFT 公开数据集极少，无法直接获取
2. **模板质量可控**: 通过精心设计的模板和多样化参数，可以生成较高质量的指令-响应对
3. **后续迭代路径**: Phase 2 产出的模板数据作为种子集，Phase 5 训练完第一版模型后可回头做 Self-Instruct 扩充

### 2.4 Train/Val 切分策略

| 方案 | 独立同分布 | 来源隔离 | 实现 | 结论 |
|------|-----------|----------|------|------|
| **随机切分** | ✅ | ❌ 同一来源可能跨 train/val | 最简单 | ✅ 选用 |
| 基于来源切分 | ⚠️ | ✅ | 中 | ❌ 来源分布不均 |
| 基于 Hash 切分 | ✅ | ❌ | 中 | 备选（确定性） |

**最终选择: 随机切分（固定 seed）**。数据源仅有 5 个，按来源切分会导致 val 集过于依赖某一来源的数据分布。固定 seed 的随机切分保证了可复现性和独立同分布。

---

## 3. 最终方案

### 3.1 模块结构

```
src/classic_chinese_llm/data/
├── __init__.py
├── collector.py
├── cleaner.py
├── deduplicator.py
├── formatter.py         # Formatter 编排器 + 内置任务模板
└── ...
```

Formatter 为单文件模块（~350 行），包含 `TaskTemplate` 数据类、8 种内置任务模板、`Formatter` 编排类。

### 3.2 核心接口设计

```python
# data/formatter.py

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# ─── 数据模型 ──────────────────────────────────────────────────────────


@dataclass
class FormattedSample:
    """格式化后的单条指令样本（ChatML 格式）。"""

    messages: list[dict[str, str]]
    # messages[0]: {"role": "system", "content": "..."}
    # messages[1]: {"role": "user", "content": "..."}
    # messages[2]: {"role": "assistant", "content": "..."}
    task_type: str       # 任务类型标识
    source_doc: str = "" # 追溯来源文档 ID（便于后续分析）

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "task_type": self.task_type,
        }

    def is_valid(self, min_response_len: int = 5) -> bool:
        """快速校验：assistant 回复不为空且长度合格。"""
        if len(self.messages) < 3:
            return False
        content = self.messages[2].get("content", "")
        return len(content.strip()) >= min_response_len


# ─── 任务模板定义 ──────────────────────────────────────────────────────


@dataclass
class TaskTemplate:
    """任务模板 —— 定义一种指令任务类型的生成规则。

    包含:
    - system_prompt: 系统提示词（塑造模型 persona）
    - instruction_template: 用户指令模板（{} 占位符填入文本）
    - response_generator: 响应生成函数 (text, metadata) → response_string
    - weight: 采样权重（用于均衡各类型占比）
    """

    task_type: str
    display_name: str
    weight: float = 1.0  # 采样权重

    # 系统提示词（支持 {era}, {genre} 等元信息占位符）
    system_prompt: str = ""

    # 用户指令模板（{text} 为必选占位符，{title} 等为可选）
    instruction_template: str = ""

    # 响应生成器: 接收 (text, metadata_dict) → 返回 response 字符串
    response_generator: Callable[[str, dict], str] | None = None


# ─── 内置系统提示词 ────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = (
    "你是一位精通中国古代文学的文言文专家。你熟读经史子集，擅长文言文阅读、"
    "翻译、创作和赏析。你的回答准确、典雅、符合古文规范。"
)

# 各任务类型的系统提示词变体
_SYSTEM_PROMPTS = {
    "translate_wen_to_bai": _SYSTEM_PROMPT_BASE + "你擅长将文言文准确翻译为流畅的现代白话文。",
    "translate_bai_to_wen": _SYSTEM_PROMPT_BASE + "你擅长将现代白话文优雅地翻译为文言文。",
    "completion": _SYSTEM_PROMPT_BASE + "你擅长根据上下文续写文言文，续写内容应风格一致、内容连贯。",
    "summarize": _SYSTEM_PROMPT_BASE + "你擅长用简洁的文言文概括文章主旨。",
    "explain_word": _SYSTEM_PROMPT_BASE + "你擅长解释文言文中的字词含义，包括本义、引申义和语境用法。",
    "grammar_analysis": _SYSTEM_PROMPT_BASE + "你擅长分析文言文的语法结构、句式和修辞手法。",
    "compose_poetry": _SYSTEM_PROMPT_BASE + "你擅长创作古典诗词，能够按照格律和意境要求进行创作。",
    "compose_prose": _SYSTEM_PROMPT_BASE + "你擅长创作文言散文，文风典雅，合乎古文规范。",
}


# ─── 内置响应生成器 ────────────────────────────────────────────────────

def _response_pass_through(text: str, metadata: dict) -> str:
    """直通响应：直接返回原文（用于续写、创作类任务）。"""
    return text


def _response_empty_placeholder(text: str, metadata: dict) -> str:
    """占位响应：生成占位文本（等待 Phase 5 后用 LLM 填充）。

    对于翻译、概括、释义等需要"生成新文本"的任务类型，
    Phase 2 使用基于原文的启发式方法生成初步响应。
    """
    return f"[待生成] 基于原文的响应，原文长度: {len(text)} 字符"


def _response_explain_word(text: str, metadata: dict) -> str:
    """词句释义的启发式响应：给出字符基本信息。"""
    chars = text.strip()
    return f"「{chars}」共 {len(chars)} 字，为文言文段落。"


# ─── 8 种内置任务模板 ──────────────────────────────────────────────────

_BUILTIN_TEMPLATES: list[TaskTemplate] = [
    # 1. 文言→白话翻译 (权重 2.0，核心任务)
    TaskTemplate(
        task_type="translate_wen_to_bai",
        display_name="文言→白话翻译",
        weight=2.0,
        system_prompt=_SYSTEM_PROMPTS["translate_wen_to_bai"],
        instruction_template="请将以下文言文翻译成现代白话文：\n\n{text}",
        response_generator=_response_empty_placeholder,
    ),

    # 2. 白话→文言翻译 (权重 1.5)
    TaskTemplate(
        task_type="translate_bai_to_wen",
        display_name="白话→文言翻译",
        weight=1.5,
        system_prompt=_SYSTEM_PROMPTS["translate_bai_to_wen"],
        instruction_template="请将以下现代白话文翻译成文言文：\n\n{text}",
        response_generator=_response_empty_placeholder,
    ),

    # 3. 文言文续写 (权重 1.5)
    TaskTemplate(
        task_type="completion",
        display_name="文言文续写",
        weight=1.5,
        system_prompt=_SYSTEM_PROMPTS["completion"],
        instruction_template="请续写以下文言文段落，保持风格和内容一致：\n\n{text}",
        response_generator=_response_pass_through,
    ),

    # 4. 主旨概括 (权重 1.0)
    TaskTemplate(
        task_type="summarize",
        display_name="主旨概括",
        weight=1.0,
        system_prompt=_SYSTEM_PROMPTS["summarize"],
        instruction_template="请用简洁的文言文概括以下段落的主旨：\n\n{text}",
        response_generator=_response_empty_placeholder,
    ),

    # 5. 词句释义 (权重 1.0)
    TaskTemplate(
        task_type="explain_word",
        display_name="词句释义",
        weight=1.0,
        system_prompt=_SYSTEM_PROMPTS["explain_word"],
        instruction_template="请解释以下文言文字词的含义和用法：\n\n{text}",
        response_generator=_response_explain_word,
    ),

    # 6. 语法分析 (权重 0.8)
    TaskTemplate(
        task_type="grammar_analysis",
        display_name="语法分析",
        weight=0.8,
        system_prompt=_SYSTEM_PROMPTS["grammar_analysis"],
        instruction_template="请分析以下文言文的句式和语法结构：\n\n{text}",
        response_generator=_response_empty_placeholder,
    ),

    # 7. 诗词创作 (权重 0.5)
    TaskTemplate(
        task_type="compose_poetry",
        display_name="诗词创作",
        weight=0.5,
        system_prompt=_SYSTEM_PROMPTS["compose_poetry"],
        instruction_template="请以「{title}」为题，创作一首七言绝句：",
        response_generator=_response_pass_through,
    ),

    # 8. 文言散文创作 (权重 0.5)
    TaskTemplate(
        task_type="compose_prose",
        display_name="文言散文创作",
        weight=0.5,
        system_prompt=_SYSTEM_PROMPTS["compose_prose"],
        instruction_template="请以「{title}」为题，写一篇简短的文言散文：",
        response_generator=_response_pass_through,
    ),
]

_DEFAULT_TOTAL_WEIGHT = sum(t.weight for t in _BUILTIN_TEMPLATES)
# = 2.0 + 1.5 + 1.5 + 1.0 + 1.0 + 0.8 + 0.5 + 0.5 = 8.8


# ─── 格式化编排器 ──────────────────────────────────────────────────────


@dataclass
class FormatterConfig:
    """格式化器配置。"""

    max_samples: int = 15000        # 最大样本数（-1 表示无限制）
    val_split: float = 0.05         # 验证集比例
    seed: int = 42                  # 随机种子（shuffle + split）
    min_response_len: int = 5       # 最小响应长度（字符）
    min_source_text_len: int = 50   # 最小源文本长度（太短的文本不适合做指令任务）


@dataclass
class FormattingStats:
    """格式化统计信息。"""

    input_docs: int = 0
    total_generated: int = 0
    total_valid: int = 0
    train_count: int = 0
    val_count: int = 0
    # 按任务类型统计
    per_task: dict[str, int] = field(default_factory=dict)


class Formatter:
    """指令数据格式化编排器。

    流程:
    1. 从去重 JSONL 读入文档
    2. 对每条文档，随机采样任务类型（按权重）
    3. 应用对应模板生成 ChatML 格式 sample
    4. 质量过滤
    5. Shuffle → Train/Val 切分 → 写 JSONL

    用法:
        formatter = Formatter(FormatterConfig(max_samples=10000))
        stats = formatter.format(input_path, output_dir)
    """

    def __init__(self, config: FormatterConfig | None = None) -> None:
        self.config = config or FormatterConfig()
        self.templates = list(_BUILTIN_TEMPLATES)
        self._rng = random.Random(self.config.seed)

    def add_template(self, template: TaskTemplate) -> None:
        """注册自定义任务模板。"""
        self.templates.append(template)

    def format(
        self,
        input_path: str | Path,
        output_dir: str | Path,
    ) -> FormattingStats:
        """执行格式化流程。

        Args:
            input_path: 输入 JSONL（Deduplicator 的输出）
            output_dir: 输出目录（生成 train.jsonl 和 val.jsonl）

        Returns:
            FormattingStats 统计信息
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = FormattingStats()

        # 1. 加载文档
        with open(input_path, encoding="utf-8") as f:
            docs = [json.loads(line.strip()) for line in f if line.strip()]

        stats.input_docs = len(docs)
        logger.info("加载 %d 篇文档用于指令数据生成", stats.input_docs)

        # 2. 过滤源文本过短的文档
        docs = [
            d for d in docs
            if len(d.get("text", "")) >= self.config.min_source_text_len
        ]
        logger.info(
            "过滤短文本: 保留 %d 篇 (min_len=%d)",
            len(docs), self.config.min_source_text_len,
        )

        # 3. 生成指令-响应对
        samples: list[FormattedSample] = []
        total_weight = sum(t.weight for t in self.templates)

        for doc in docs:
            # 按权重随机采样任务类型
            r = self._rng.random() * total_weight
            cumulative = 0.0
            selected_template = self.templates[-1]  # fallback
            for tpl in self.templates:
                cumulative += tpl.weight
                if r <= cumulative:
                    selected_template = tpl
                    break

            sample = self._apply_template(doc, selected_template)
            if sample and sample.is_valid(self.config.min_response_len):
                samples.append(sample)
                stats.total_valid += 1

            stats.total_generated += 1

            # 达到最大样本数则提前结束
            if self.config.max_samples > 0 and stats.total_valid >= self.config.max_samples:
                break

        # 4. Shuffle
        self._rng.shuffle(samples)

        # 5. Train/Val 切分
        val_size = max(1, int(len(samples) * self.config.val_split))
        val_samples = samples[:val_size]
        train_samples = samples[val_size:]

        stats.train_count = len(train_samples)
        stats.val_count = len(val_samples)

        # 6. 按任务类型统计
        for s in samples:
            stats.per_task[s.task_type] = stats.per_task.get(s.task_type, 0) + 1

        # 7. 写出
        self._write_jsonl(train_samples, output_dir / "train.jsonl")
        self._write_jsonl(val_samples, output_dir / "val.jsonl")

        logger.info(
            "格式化完成: %d 条样本 (train=%d, val=%d), %d 种任务类型",
            stats.total_valid, stats.train_count, stats.val_count,
            len(stats.per_task),
        )
        for task, count in sorted(stats.per_task.items()):
            logger.info("  %s: %d (%.1f%%)", task, count, 100 * count / max(stats.total_valid, 1))

        return stats

    def _apply_template(
        self,
        doc: dict,
        template: TaskTemplate,
    ) -> FormattedSample | None:
        """将文档应用于任务模板，生成 ChatML 样本。"""
        text = doc.get("text", "")
        if not text.strip():
            return None

        # 填充 instruction 模板
        title = doc.get("title", "未命名")
        metadata = {
            "title": title,
            "author": doc.get("author", ""),
            "era": doc.get("era", ""),
            "genre": doc.get("genre", ""),
        }

        # 支持 {text} 和 {title} 占位符
        try:
            instruction = template.instruction_template.format(
                text=text,
                title=title,
            )
            system = template.system_prompt.format(**metadata)
        except KeyError:
            # 模板中有不支持的占位符时跳过
            return None

        # 生成响应
        if template.response_generator:
            response = template.response_generator(text, metadata)
        else:
            response = text  # fallback: 直通原文

        return FormattedSample(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            task_type=template.task_type,
            source_doc=doc.get("source", ""),
        )

    @staticmethod
    def _write_jsonl(samples: list[FormattedSample], path: Path) -> None:
        """写出 ChatML JSONL 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
```

### 3.3 使用示例

```python
from classic_chinese_llm.data.formatter import Formatter, FormatterConfig

config = FormatterConfig(
    max_samples=15000,
    val_split=0.05,
    seed=42,
    min_source_text_len=50,
)

formatter = Formatter(config)
stats = formatter.format(
    "data/processed/deduplicated.jsonl",
    "data/processed/instructions/",
)
print(f"Train: {stats.train_count}, Val: {stats.val_count}")
print(f"Task distribution: {stats.per_task}")
```

### 3.4 输出格式

生成的 `train.jsonl` / `val.jsonl` 每行为：

```json
{
  "messages": [
    {"role": "system", "content": "你是一位精通中国古代文学的文言文专家..."},
    {"role": "user", "content": "请将以下文言文翻译成现代白话文：\n\n子曰：学而时习之，不亦说乎？"},
    {"role": "assistant", "content": "[待生成] 基于原文的响应，原文长度: 17 字符"}
  ],
  "task_type": "translate_wen_to_bai"
}
```

下游 SFT 训练代码中：
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
# → "<|system|>你是一位...<|user|>请将以下...<|assistant|>[待生成]..."
```

---

## 4. 关键技术点

### 4.1 模板占位符预校验

模板中包含 `{text}`、`{title}` 等 Python format 占位符。如果文档中缺失对应字段（如某些来源没有 `title`），`str.format()` 会抛出 `KeyError`。

处理策略：`_apply_template()` 中 `try/except KeyError` 捕获后返回 `None` 跳过该条。这避免了因少数文档缺失元信息而导致整个格式化流程中断。

### 4.2 响应生成器的阶段演进策略

这是整个 Formatter 设计中最关键的决策点——Phase 2 阶段没有可用的 LLM，无法生成高质量的"翻译"、"概括"类响应。

| 任务类型 | Phase 2 策略 | Phase 5 后升级策略 |
|----------|-------------|-------------------|
| 续写类（completion, 创作） | 直通原文（原文=response） | 模型生成（用 checkpoint 做推理） |
| 翻译类（文言↔白话） | 占位文本 `[待生成]` | 模型生成或人工标注 |
| 分析类（释义、语法） | 占位文本 `[待生成]` | 模型生成或人工标注 |

这种策略允许 Phase 2 立即产出一个可用的训练集格式（用于验证 pipeline 全流程），Phase 5 拿到训练好的模型后再用 Self-Instruct 方式大规模扩充高质量响应。

### 4.3 任务类型权重设计

权重分配基于两个因素：(1) 下游对话场景的实用度；(2) 模板响应的可靠度。

```
文言→白话翻译:  2.0  ← 最实用（对话场景最常见需求）
白话→文言翻译:  1.5  ← 次实用
文言文续写:     1.5  ← 响应可靠（直通原文）
主旨概括:       1.0  ← 有用但响应需后续增强
词句释义:       1.0  ← 有用但响应需后续增强
语法分析:       0.8  ← 专业场景，小权重
诗词创作:       0.5  ← 创意类，小权重
文言散文创作:   0.5  ← 创意类，小权重
─────────────────────
总权重:         8.8
```

实际采样时，每条文档按 `random() * total_weight` 的概率区间分配任务类型。

### 4.4 Seed 固定的可复现性

Formatter 中所有随机操作（任务类型采样、shuffle、train/val 切分）均使用 `self._rng = random.Random(seed)`。这意味着：

- 相同 seed + 相同输入文档 → 完全相同的输出
- CI 中可验证格式化输出的 checksum
- 不同 seed 产生不同的任务分配和切分（用于数据多样性实验）

### 4.5 质量过滤的两个层次

Formatter 在两个层次上确保输出质量：

**文档级过滤**（格式化前）：
- `min_source_text_len=50`：过滤过短文档，它们不适合做指令任务的基础

**样本级过滤**（格式化后）：
- `min_response_len=5`：过滤响应过短的样本（占位文本也有足够长度）
- `messages` 长度检查：确保 system/user/assistant 三个角色齐全

这两个层次的过滤在 `format()` 方法中串联执行，统计信息中记录每层的过滤数量。

### 4.6 与 SFT DataConfig 的集成

`FormatterConfig` 中的 `max_samples` 和 `val_split` 与 SFT 的 `DataConfig` 对应：

```python
# config/settings.py 中已有的 DataConfig
class DataConfig(BaseModel):
    max_samples: int = 15000
    val_split: float = 0.05

# formatter.py 中与之对应
config = FormatterConfig(
    max_samples=data_config.max_samples,
    val_split=data_config.val_split,
)
```

Phase 5 SFT 训练脚本可以直接使用 `FormatterConfig` 的参数来确保数据量与配置一致。

---

## 5. 与其他模块的关系

```
Config ─── 被依赖 ───> Formatter (DataConfig.max_samples, val_split)
Utils  ─── 被依赖 ───> Formatter (logging)

Deduplicator ──→ 输出 deduplicated.jsonl ──→ Formatter ──→ 输出 train.jsonl + val.jsonl
                                                                      │
                                                                      ↓
                                                              SFT Trainer (Phase 5)
```

Formatter 是 Phase 2 数据管道的**终端节点**。它的输出不流向 Phase 3 (Tokenizer) 或 Phase 4 (Pretrain)，而是直接流向 Phase 5 (SFT)。

Phase 4 的预训练直接使用 Deduplicator 输出的 `deduplicated.jsonl`（去重后的原始文言文文本），不需要经过 Formatter 处理。

### 数据管道完整流程

```
[5 个数据源]
     │
     ▼
 Collector  ──→  data/processed/collected.jsonl     (~18亿字符)
     │
     ▼
 Cleaner    ──→  data/processed/cleaned.jsonl        (~15亿字符, 清洗损耗~15%)
     │
     ▼
 Deduplicator ──→ data/processed/deduplicated.jsonl  (~14亿字符, 去重损耗~10-20%)
     │
     ├──→ Pretrain (Phase 4) — 直接使用去重后的纯文本
     │
     └──→ Formatter ──→ data/processed/instructions/train.jsonl  (10K-15K 条)
                    ──→ data/processed/instructions/val.jsonl    (500-750 条)
                              │
                              └──→ SFT (Phase 5)
```

---

## 6. 验证清单

- [ ] `FormattedSample.is_valid()` 对 3 条完整 messages 返回 True
- [ ] `FormattedSample.is_valid()` 对空 assistant 回复返回 False
- [ ] 所有 8 种内置模板的 `instruction_template` 中 `{text}` 占位符正确
- [ ] 同一 seed + 同一输入两次运行输出完全一致（可复现性）
- [ ] `max_samples=100` 时实际生成 ≤100 条
- [ ] `val_split=0.1` 时验证集占比约为 10%
- [ ] 输出 JSONL 每行可通过 `json.loads` 解析
- [ ] 输出的 ChatML 格式可通过 `tokenizer.apply_chat_template()` 序列化
- [ ] 各任务类型的统计占比大致符合权重比例
- [ ] `min_source_text_len=50` 时，所有输入文档的 text 字段 ≥50 字符
- [ ] 模板占位符缺失时 `_apply_template` 返回 None 而非崩溃
