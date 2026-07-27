"""数据格式化器。

将去重后的文言文段落通过任务模板转换为 ChatML 格式的指令-响应对，
用于下游 SFT 微调训练。
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


# ─── 数据模型 ──────────────────────────────────────────────────────────


@dataclass
class FormattedSample:
    """格式化后的单条指令样本（ChatML 格式）。"""

    messages: list[dict[str, str]]
    task_type: str
    source_doc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"messages": self.messages, "task_type": self.task_type}

    def is_valid(self, min_response_len: int = 5) -> bool:
        """快速校验：assistant 回复不为空且长度合格。"""
        if len(self.messages) < 3:
            return False
        content = self.messages[2].get("content", "")
        return len(content.strip()) >= min_response_len


# ─── 任务模板定义 ──────────────────────────────────────────────────────


@dataclass
class TaskTemplate:
    """任务模板 —— 定义一种指令任务类型的生成规则。"""

    task_type: str
    display_name: str
    weight: float = 1.0
    system_prompt: str = ""
    instruction_template: str = ""
    response_generator: Callable[[str, dict[str, str]], str] | None = None


# ─── 内置系统提示词 ────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = (
    "你是一位精通中国古代文学的文言文专家。你熟读经史子集，擅长文言文阅读、"
    "翻译、创作和赏析。你的回答准确、典雅、符合古文规范。"
)

_SYSTEM_PROMPTS: dict[str, str] = {
    "translate_wen_to_bai": (_SYSTEM_PROMPT_BASE + "你擅长将文言文准确翻译为流畅的现代白话文。"),
    "translate_bai_to_wen": (_SYSTEM_PROMPT_BASE + "你擅长将现代白话文优雅地翻译为文言文。"),
    "completion": (
        _SYSTEM_PROMPT_BASE + "你擅长根据上下文续写文言文，续写内容应风格一致、内容连贯。"
    ),
    "summarize": (_SYSTEM_PROMPT_BASE + "你擅长用简洁的文言文概括文章主旨。"),
    "explain_word": (
        _SYSTEM_PROMPT_BASE + "你擅长解释文言文中的字词含义，包括本义、引申义和语境用法。"
    ),
    "grammar_analysis": (_SYSTEM_PROMPT_BASE + "你擅长分析文言文的语法结构、句式和修辞手法。"),
    "compose_poetry": (
        _SYSTEM_PROMPT_BASE + "你擅长创作古典诗词，能够按照格律和意境要求进行创作。"
    ),
    "compose_prose": (_SYSTEM_PROMPT_BASE + "你擅长创作文言散文，文风典雅，合乎古文规范。"),
}


# ─── 内置响应生成器 ────────────────────────────────────────────────────


def _response_pass_through(text: str, _metadata: dict[str, str]) -> str:
    """直通响应：直接返回原文。"""
    return text


def _response_empty_placeholder(text: str, _metadata: dict[str, str]) -> str:
    """占位响应：等待 Phase 5 后用 LLM 填充。"""
    return f"[待生成] 基于原文的响应，原文长度: {len(text)} 字符"


def _response_explain_word(text: str, _metadata: dict[str, str]) -> str:
    """词句释义的启发式响应。"""
    chars = text.strip()
    return f"「{chars}」共 {len(chars)} 字，为文言文段落。"


# ─── 8 种内置任务模板 ──────────────────────────────────────────────────

_BUILTIN_TEMPLATES: list[TaskTemplate] = [
    TaskTemplate(
        task_type="translate_wen_to_bai",
        display_name="文言→白话翻译",
        weight=2.0,
        system_prompt=_SYSTEM_PROMPTS["translate_wen_to_bai"],
        instruction_template="请将以下文言文翻译成现代白话文：\n\n{text}",
        response_generator=_response_empty_placeholder,
    ),
    TaskTemplate(
        task_type="translate_bai_to_wen",
        display_name="白话→文言翻译",
        weight=1.5,
        system_prompt=_SYSTEM_PROMPTS["translate_bai_to_wen"],
        instruction_template="请将以下现代白话文翻译成文言文：\n\n{text}",
        response_generator=_response_empty_placeholder,
    ),
    TaskTemplate(
        task_type="completion",
        display_name="文言文续写",
        weight=1.5,
        system_prompt=_SYSTEM_PROMPTS["completion"],
        instruction_template="请续写以下文言文段落，保持风格和内容一致：\n\n{text}",
        response_generator=_response_pass_through,
    ),
    TaskTemplate(
        task_type="summarize",
        display_name="主旨概括",
        weight=1.0,
        system_prompt=_SYSTEM_PROMPTS["summarize"],
        instruction_template="请用简洁的文言文概括以下段落的主旨：\n\n{text}",
        response_generator=_response_empty_placeholder,
    ),
    TaskTemplate(
        task_type="explain_word",
        display_name="词句释义",
        weight=1.0,
        system_prompt=_SYSTEM_PROMPTS["explain_word"],
        instruction_template="请解释以下文言文字词的含义和用法：\n\n{text}",
        response_generator=_response_explain_word,
    ),
    TaskTemplate(
        task_type="grammar_analysis",
        display_name="语法分析",
        weight=0.8,
        system_prompt=_SYSTEM_PROMPTS["grammar_analysis"],
        instruction_template="请分析以下文言文的句式和语法结构：\n\n{text}",
        response_generator=_response_empty_placeholder,
    ),
    TaskTemplate(
        task_type="compose_poetry",
        display_name="诗词创作",
        weight=0.5,
        system_prompt=_SYSTEM_PROMPTS["compose_poetry"],
        instruction_template="请以「{title}」为题，创作一首七言绝句：",
        response_generator=_response_pass_through,
    ),
    TaskTemplate(
        task_type="compose_prose",
        display_name="文言散文创作",
        weight=0.5,
        system_prompt=_SYSTEM_PROMPTS["compose_prose"],
        instruction_template="请以「{title}」为题，写一篇简短的文言散文：",
        response_generator=_response_pass_through,
    ),
]


# ─── 配置 ──────────────────────────────────────────────────────────────


@dataclass
class FormatterConfig:
    """格式化器配置。"""

    max_samples: int = 15000
    val_split: float = 0.05
    seed: int = 42
    min_response_len: int = 5
    min_source_text_len: int = 50


# ─── 统计 ──────────────────────────────────────────────────────────────


@dataclass
class FormattingStats:
    """格式化统计信息。"""

    input_docs: int = 0
    total_generated: int = 0
    total_valid: int = 0
    train_count: int = 0
    val_count: int = 0
    per_task: dict[str, int] = field(default_factory=dict)


# ─── 格式化编排器 ──────────────────────────────────────────────────────


class Formatter:
    """指令数据格式化编排器。

    流程:
    1. 从去重 JSONL 读入文档
    2. 对每条文档，按权重随机采样任务类型
    3. 应用对应模板生成 ChatML 格式 sample
    4. 质量过滤 → Shuffle → Train/Val 切分 → 写 JSONL

    用法:
        formatter = Formatter(FormatterConfig(max_samples=10000))
        stats = formatter.format(input_path, output_dir)
    """

    def __init__(self, config: FormatterConfig | None = None) -> None:
        self.config = config or FormatterConfig()
        self.templates: list[TaskTemplate] = list(_BUILTIN_TEMPLATES)
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
        docs = [d for d in docs if len(d.get("text", "")) >= self.config.min_source_text_len]
        logger.info(
            "过滤短文本: 保留 %d 篇 (min_len=%d)",
            len(docs),
            self.config.min_source_text_len,
        )

        # 3. 生成指令-响应对
        samples: list[FormattedSample] = []
        total_weight = sum(t.weight for t in self.templates)

        for doc in docs:
            template = self._sample_template(total_weight)
            sample = self._apply_template(doc, template)
            if sample and sample.is_valid(self.config.min_response_len):
                samples.append(sample)
                stats.total_valid += 1

            stats.total_generated += 1

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
            stats.total_valid,
            stats.train_count,
            stats.val_count,
            len(stats.per_task),
        )
        for task, count in sorted(stats.per_task.items()):
            logger.info(
                "  %s: %d (%.1f%%)",
                task,
                count,
                100 * count / max(stats.total_valid, 1),
            )

        return stats

    def _sample_template(self, total_weight: float) -> TaskTemplate:
        """按权重随机采样任务类型。"""
        r = self._rng.random() * total_weight
        cumulative = 0.0
        for tpl in self.templates:
            cumulative += tpl.weight
            if r <= cumulative:
                return tpl
        return self.templates[-1]

    def _apply_template(
        self, doc: dict[str, Any], template: TaskTemplate
    ) -> FormattedSample | None:
        """将文档应用于任务模板，生成 ChatML 样本。"""
        text: str = doc.get("text", "")
        if not text.strip():
            return None

        title = doc.get("title", "未命名")
        metadata = {
            "title": title,
            "author": doc.get("author", ""),
            "era": doc.get("era", ""),
            "genre": doc.get("genre", ""),
        }

        # 填充模板占位符
        try:
            instruction = template.instruction_template.format(text=text, title=title)
            system = template.system_prompt.format(**metadata)
        except KeyError:
            return None

        # 生成响应
        if template.response_generator:
            response = template.response_generator(text, metadata)
        else:
            response = text

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
