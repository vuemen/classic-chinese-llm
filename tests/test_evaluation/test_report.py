"""evaluation.report 模块的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from classic_chinese_llm.evaluation.config import EvalConfig
from classic_chinese_llm.evaluation.report import EvalReport, EvalSample


class TestEvalSample:
    """EvalSample 数据类测试。"""

    def test_creation(self) -> None:
        """基本创建测试。"""
        sample = EvalSample(
            prompt="请翻译：学而时习之",
            reference="学而时习之",
            prediction="学而时习之不亦说乎",
            metrics={"bleu": 0.75, "char_accuracy": 0.8},
        )
        assert sample.prompt == "请翻译：学而时习之"
        assert sample.metrics["bleu"] == 0.75


class TestEvalReport:
    """EvalReport 测试。"""

    def _make_report(self, tmp_path: Path) -> EvalReport:
        """创建测试用的 EvalReport。"""
        config = EvalConfig(max_samples=10, output_dir=tmp_path)
        samples = [
            EvalSample(
                prompt="Q1",
                reference="A1",
                prediction="A1_pred",
                metrics={"bleu": 0.8},
            ),
            EvalSample(
                prompt="Q2",
                reference="A2",
                prediction="A2_pred",
                metrics={"bleu": 0.6},
            ),
        ]
        return EvalReport(
            config=config,
            samples=samples,
            aggregate_metrics={"bleu": 0.7, "rouge_l": 0.65},
            timestamp="2026-07-28T00:00:00",
            model_info={"model_name": "classical-chinese-llm", "params": 157000000},
        )

    def test_summary_non_empty(self, temp_dir: Path) -> None:
        """summary() 应返回非空字符串。"""
        report = self._make_report(temp_dir)
        text = report.summary()
        assert len(text) > 0
        assert "Aggregate Metrics" in text or "评估" in text or "BLEU" in text

    def test_to_json(self, temp_dir: Path) -> None:
        """to_json 应生成有效 JSON 文件。"""
        report = self._make_report(temp_dir)
        output_path = temp_dir / "eval_report.json"
        report.to_json(output_path)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["aggregate_metrics"]["bleu"] == 0.7
        assert len(data["samples"]) == 2
        assert data["model_info"]["params"] == 157000000

    def test_json_contains_required_fields(self, temp_dir: Path) -> None:
        """JSON 输出应包含所有必要字段。"""
        report = self._make_report(temp_dir)
        output_path = temp_dir / "report.json"
        report.to_json(output_path)

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        for key in ["config", "samples", "aggregate_metrics", "timestamp", "model_info"]:
            assert key in data, f"缺少必要字段: {key}"
