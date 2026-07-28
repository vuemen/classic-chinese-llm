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

    def to_markdown(self, path: Path) -> None:
        """将报告导出为 Markdown 文件。

        生成包含评测摘要、指标表格、逐样本详情的格式化 Markdown 报告。

        Args:
            path: 输出 Markdown 文件路径。
        """
        lines: list[str] = []
        lines.append("# 文言文 LLM 评测报告")
        lines.append("")
        lines.append(f"**评测时间**: {self.timestamp}  ")
        lines.append(f"**样本数量**: {len(self.samples)}  ")
        lines.append(f"**模型**: {self.model_info.get('model_class', 'N/A')}  ")
        lines.append(f"**参数量**: {self.model_info.get('total_params', 0):,}  ")
        if self.model_info.get("checkpoint_name"):
            lines.append(f"**Checkpoint**: {self.model_info['checkpoint_name']}  ")
        if self.model_info.get("dataset_name"):
            lines.append(f"**数据集**: {self.model_info['dataset_name']}  ")
        lines.append("")

        # ─── 聚合指标表 ───────────────────────────────────────────────
        lines.append("## 聚合指标")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        for name, value in self.aggregate_metrics.items():
            lines.append(f"| {name} | {value:.4f} |")
        lines.append("")

        # ─── 分布统计 ─────────────────────────────────────────────────
        lines.append("## 指标分布")
        lines.append("")
        self._append_distribution(lines)

        # ─── 最佳/最差样本 ────────────────────────────────────────────
        lines.append("## 极端样本")
        lines.append("")
        self._append_extreme_samples(lines)

        # ─── 逐样本详情 ───────────────────────────────────────────────
        lines.append("## 逐样本详情")
        lines.append("")
        self._append_sample_table(lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _append_distribution(self, lines: list[str]) -> None:
        """追加指标分布统计（min / max / mean / std）。"""
        if not self.samples:
            lines.append("无样本。")
            lines.append("")
            return

        # 收集所有逐样本指标名
        metric_names: set[str] = set()
        for s in self.samples:
            metric_names.update(s.metrics.keys())

        if not metric_names:
            lines.append("无逐样本指标。")
            lines.append("")
            return

        lines.append("| 指标 | 均值 | 标准差 | 最小值 | 最大值 |")
        lines.append("|------|------|--------|--------|--------|")
        for name in sorted(metric_names):
            values = [s.metrics.get(name) for s in self.samples]
            valid = [v for v in values if v is not None]
            if not valid:
                continue
            mean_v = sum(valid) / len(valid)
            std_v = (
                (sum((v - mean_v) ** 2 for v in valid) / len(valid)) ** 0.5
                if len(valid) > 1
                else 0.0
            )
            min_v = min(valid)
            max_v = max(valid)
            lines.append(f"| {name} | {mean_v:.4f} | {std_v:.4f} | {min_v:.4f} | {max_v:.4f} |")
        lines.append("")

    def _append_extreme_samples(self, lines: list[str]) -> None:
        """追加最佳和最差样本（按总分或第一个指标排序）。"""
        if not self.samples:
            lines.append("无样本。")
            lines.append("")
            return

        # 确定排序指标: 优先用 classical_总分, 否则用第一个逐样本指标
        sort_key = "classical_总分"
        if self.samples and sort_key not in self.samples[0].metrics:
            if self.samples[0].metrics:
                sort_key = next(iter(self.samples[0].metrics))
            else:
                lines.append("无逐样本指标，无法排序。")
                lines.append("")
                return

        ranked = sorted(
            self.samples,
            key=lambda s: s.metrics.get(sort_key, 0.0),
            reverse=True,
        )

        # 最佳 3 条
        lines.append(f"### 最佳样本（按 {sort_key}，前 3）")
        lines.append("")
        for i, sample in enumerate(ranked[:3], 1):
            lines.append(f"**#{i}** (score: {sample.metrics.get(sort_key, 0):.4f})")
            lines.append(
                f"- Prompt: {sample.prompt[:120]}{'...' if len(sample.prompt) > 120 else ''}"
            )
            lines.append(
                f"- Reference: {sample.reference[:120]}{'...' if len(sample.reference) > 120 else ''}"
            )
            lines.append(
                f"- Prediction: {sample.prediction[:120]}{'...' if len(sample.prediction) > 120 else ''}"
            )
            lines.append("")

        # 最差 3 条
        lines.append(f"### 最差样本（按 {sort_key}，后 3）")
        lines.append("")
        for i, sample in enumerate(ranked[-3:], 1):
            lines.append(f"**#{i}** (score: {sample.metrics.get(sort_key, 0):.4f})")
            lines.append(
                f"- Prompt: {sample.prompt[:120]}{'...' if len(sample.prompt) > 120 else ''}"
            )
            lines.append(
                f"- Reference: {sample.reference[:120]}{'...' if len(sample.reference) > 120 else ''}"
            )
            lines.append(
                f"- Prediction: {sample.prediction[:120]}{'...' if len(sample.prediction) > 120 else ''}"
            )
            lines.append("")

    def _append_sample_table(self, lines: list[str]) -> None:
        """追加逐样本详情表格。"""
        if not self.samples:
            lines.append("无样本。")
            lines.append("")
            return

        # 收集所有指标列
        metric_cols: set[str] = set()
        for s in self.samples:
            metric_cols.update(s.metrics.keys())
        sorted_cols = sorted(metric_cols)

        # 表头
        header = "| # | Prompt | Reference | Prediction |"
        sep = "|---|--------|-----------|------------|"
        if sorted_cols:
            for col in sorted_cols:
                short_col = col.replace("classical_", "c_")
                header += f" {short_col} |"
                sep += "------|"
        lines.append(header)
        lines.append(sep)

        # 数据行
        for idx, s in enumerate(self.samples, 1):
            prompt_short = s.prompt[:40] + "..." if len(s.prompt) > 40 else s.prompt
            ref_short = s.reference[:40] + "..." if len(s.reference) > 40 else s.reference
            pred_short = s.prediction[:40] + "..." if len(s.prediction) > 40 else s.prediction
            row = f"| {idx} | {prompt_short} | {ref_short} | {pred_short} |"
            for col in sorted_cols:
                val = s.metrics.get(col)
                row += f" {val:.3f} |" if val is not None else " - |"
            lines.append(row)
        lines.append("")

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
