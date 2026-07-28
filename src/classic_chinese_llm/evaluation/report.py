"""评测报告 —— EvalSample 和 EvalReport 数据结构。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from classic_chinese_llm.evaluation.config import EvalConfig


@dataclass
class EvalSample:
    """单条评测样本。

    Attributes:
        prompt: 用户输入/prompt 文本。
        reference: 期望输出/参考文本。
        prediction: 模型生成的文本。
        metrics: 该样本的各项指标值。
    """

    prompt: str
    reference: str
    prediction: str
    metrics: dict[str, float]


@dataclass
class EvalReport:
    """评测报告。

    包含所有评测样本及聚合指标，支持导出为 JSON 文件和生成
    人类可读的文本总结。

    Attributes:
        config: 评测配置。
        samples: 所有评测样本。
        aggregate_metrics: 聚合的指标值 (corpus-level)。
        timestamp: 评测时间戳 (ISO 8601)。
        model_info: 模型元信息 (名称、参数量等)。
    """

    config: EvalConfig
    samples: list[EvalSample]
    aggregate_metrics: dict[str, float]
    timestamp: str
    model_info: dict[str, Any]

    def to_json(self, path: Path) -> None:
        """将报告序列化为 JSON 文件。

        Args:
            path: 输出 JSON 文件路径。
        """
        data: dict[str, Any] = {
            "config": {
                "max_samples": self.config.max_samples,
                "metrics": self.config.metrics,
            },
            "samples": [asdict(s) for s in self.samples],
            "aggregate_metrics": self.aggregate_metrics,
            "timestamp": self.timestamp,
            "model_info": self.model_info,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """生成人类可读的文本总结。

        Returns:
            str: 格式化的评测摘要文本。
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  文言文 LLM 评测报告")
        lines.append("=" * 60)
        lines.append(f"  评测时间: {self.timestamp}")
        lines.append(f"  样本数量: {len(self.samples)}")
        lines.append(f"  模型信息: {self.model_info}")
        lines.append("-" * 60)
        lines.append("  Aggregate Metrics:")
        for name, value in self.aggregate_metrics.items():
            lines.append(f"    {name:20s}: {value:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    @classmethod
    def create(
        cls,
        config: EvalConfig,
        samples: list[EvalSample],
        aggregate_metrics: dict[str, float],
        model_info: dict[str, Any] | None = None,
    ) -> EvalReport:
        """工厂方法：创建评测报告（自动填充时间戳）。

        Args:
            config: 评测配置。
            samples: 评测样本列表。
            aggregate_metrics: 聚合指标。
            model_info: 模型元信息。

        Returns:
            EvalReport 实例。
        """
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return cls(
            config=config,
            samples=samples,
            aggregate_metrics=aggregate_metrics,
            timestamp=timestamp,
            model_info=model_info or {},
        )
