"""评测配置 —— EvalConfig 数据类。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from classic_chinese_llm.model.generation import GenerationConfig


@dataclass
class EvalConfig:
    """评测运行配置。

    Attributes:
        max_samples: 评测样本上限（截断大数据集以加速评测）。
        metrics: 启用的指标列表（["perplexity", "bleu", "rouge_l", "char_accuracy"]）。
        generation: 生成参数配置（默认 temperature=0 确定性生成）。
        output_dir: 报告输出目录（None 表示仅输出到终端）。
    """

    max_samples: int = 500
    metrics: list[str] = field(
        default_factory=lambda: ["perplexity", "bleu", "rouge_l", "char_accuracy"]
    )
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """验证配置合法性。"""
        valid_metrics = {"perplexity", "bleu", "rouge_l", "char_accuracy"}
        for m in self.metrics:
            if m not in valid_metrics:
                raise ValueError(f"无效的指标: {m}。有效选项: {sorted(valid_metrics)}")
