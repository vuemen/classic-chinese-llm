"""评测配置 —— EvalConfig 数据类。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from classic_chinese_llm.model.generation import GenerationConfig

if TYPE_CHECKING:
    pass


# 有效指标列表（包含文言文专用评分）
VALID_METRICS = {"perplexity", "bleu", "rouge_l", "char_accuracy", "classical_chinese_score"}


@dataclass
class EvalConfig:
    """评测运行配置。

    Attributes:
        max_samples: 评测样本上限（截断大数据集以加速评测）。
        metrics: 启用的指标列表。
        generation: 生成参数配置（默认 temperature=0 确定性生成）。
        output_dir: 报告输出目录（None 表示仅输出到终端）。
        chat_template: ChatML 模板名称（用于指令模型评测，None 表示不包装）。
        checkpoint_name: 模型 checkpoint 名称（用于报告标识）。
        dataset_name: 评测数据集名称（用于报告标识）。
    """

    max_samples: int = 500
    metrics: list[str] = field(
        default_factory=lambda: [
            "perplexity",
            "bleu",
            "rouge_l",
            "char_accuracy",
            "classical_chinese_score",
        ]
    )
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output_dir: Path | None = None
    chat_template: str | None = None
    checkpoint_name: str = ""
    dataset_name: str = ""

    def __post_init__(self) -> None:
        """验证配置合法性。"""
        for m in self.metrics:
            if m not in VALID_METRICS:
                raise ValueError(f"无效的指标: {m}。有效选项: {sorted(VALID_METRICS)}")
