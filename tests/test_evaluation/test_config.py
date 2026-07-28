"""EvalConfig 评测配置测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from classic_chinese_llm.evaluation.config import VALID_METRICS, EvalConfig
from classic_chinese_llm.model.generation import GenerationConfig


class TestValidMetrics:
    """VALID_METRICS 常量测试。"""

    def test_contains_all_five_metrics(self) -> None:
        """VALID_METRICS 包含全部 5 个指标。"""
        assert len(VALID_METRICS) == 5
        assert "perplexity" in VALID_METRICS
        assert "bleu" in VALID_METRICS
        assert "rouge_l" in VALID_METRICS
        assert "char_accuracy" in VALID_METRICS
        assert "classical_chinese_score" in VALID_METRICS

    def test_is_a_set(self) -> None:
        """VALID_METRICS 是 set 类型，支持快速成员检查。"""
        assert isinstance(VALID_METRICS, set)

    def test_classical_chinese_score_in_valid_metrics(self) -> None:
        """文言文专用评分指标在有效指标列表中。"""
        assert "classical_chinese_score" in VALID_METRICS


class TestEvalConfigDefaults:
    """EvalConfig 默认值测试。"""

    def test_default_max_samples(self) -> None:
        """默认 max_samples 为 500。"""
        cfg = EvalConfig()
        assert cfg.max_samples == 500

    def test_default_metrics_contains_all_five(self) -> None:
        """默认启用全部 5 个评测指标。"""
        cfg = EvalConfig()
        assert len(cfg.metrics) == 5
        assert "perplexity" in cfg.metrics
        assert "bleu" in cfg.metrics
        assert "rouge_l" in cfg.metrics
        assert "char_accuracy" in cfg.metrics
        assert "classical_chinese_score" in cfg.metrics

    def test_default_generation_config(self) -> None:
        """默认生成配置为 GenerationConfig 实例。"""
        cfg = EvalConfig()
        assert isinstance(cfg.generation, GenerationConfig)
        # 评测时通常用确定性生成
        assert cfg.generation.temperature == 1.0

    def test_default_output_dir_is_none(self) -> None:
        """默认 output_dir 为 None（仅输出到终端）。"""
        cfg = EvalConfig()
        assert cfg.output_dir is None

    def test_default_chat_template_is_none(self) -> None:
        """默认 chat_template 为 None。"""
        cfg = EvalConfig()
        assert cfg.chat_template is None

    def test_default_checkpoint_name_is_empty(self) -> None:
        """默认 checkpoint_name 为空字符串。"""
        cfg = EvalConfig()
        assert cfg.checkpoint_name == ""

    def test_default_dataset_name_is_empty(self) -> None:
        """默认 dataset_name 为空字符串。"""
        cfg = EvalConfig()
        assert cfg.dataset_name == ""


class TestEvalConfigCustomValues:
    """EvalConfig 自定义参数测试。"""

    def test_custom_max_samples(self) -> None:
        """可自定义 max_samples。"""
        cfg = EvalConfig(max_samples=100)
        assert cfg.max_samples == 100

    def test_custom_metrics_subset(self) -> None:
        """仅启用指定指标子集。"""
        cfg = EvalConfig(metrics=["perplexity", "bleu"])
        assert cfg.metrics == ["perplexity", "bleu"]

    def test_classical_chinese_score_as_valid_metric(self) -> None:
        """classical_chinese_score 可以作为有效指标使用。"""
        cfg = EvalConfig(metrics=["classical_chinese_score"])
        assert cfg.metrics == ["classical_chinese_score"]

    def test_custom_chat_template(self) -> None:
        """可设置 chat_template。"""
        cfg = EvalConfig(chat_template="classical_chinese_v1")
        assert cfg.chat_template == "classical_chinese_v1"

    def test_custom_checkpoint_name(self) -> None:
        """可设置 checkpoint_name。"""
        cfg = EvalConfig(checkpoint_name="sft_best_v2")
        assert cfg.checkpoint_name == "sft_best_v2"

    def test_custom_dataset_name(self) -> None:
        """可设置 dataset_name。"""
        cfg = EvalConfig(dataset_name="classical_qa_test")
        assert cfg.dataset_name == "classical_qa_test"

    def test_custom_output_dir(self, temp_dir: Path) -> None:
        """可设置 output_dir。"""
        out_dir = temp_dir / "eval_reports"
        cfg = EvalConfig(output_dir=out_dir)
        assert cfg.output_dir == out_dir

    def test_custom_generation_config_overrides_defaults(self) -> None:
        """自定义生成配置覆盖默认值。"""
        gen_cfg = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            do_sample=False,
        )
        cfg = EvalConfig(generation=gen_cfg)
        assert cfg.generation.max_new_tokens == 512
        assert cfg.generation.temperature == 0.7
        assert cfg.generation.do_sample is False


class TestEvalConfigValidation:
    """EvalConfig 校验测试。"""

    def test_invalid_metric_raises_valueerror(self) -> None:
        """无效的指标名称抛出 ValueError。"""
        with pytest.raises(ValueError, match="无效的指标"):
            EvalConfig(metrics=["invalid_metric"])

    def test_invalid_metric_error_message_contains_valid_options(self) -> None:
        """报错信息包含有效选项列表。"""
        with pytest.raises(ValueError, match="有效选项"):
            EvalConfig(metrics=["bad_metric"])

        # 验证错误消息中包含所有有效指标
        with pytest.raises(ValueError) as exc_info:
            EvalConfig(metrics=["nonexistent"])
        err_msg = str(exc_info.value)
        for valid_metric in VALID_METRICS:
            assert valid_metric in err_msg

    def test_partial_invalid_metrics_raises(self) -> None:
        """指标列表中只要有一个无效就报错。"""
        with pytest.raises(ValueError):
            EvalConfig(metrics=["perplexity", "invalid", "bleu"])

    def test_all_valid_metrics_no_error(self) -> None:
        """全部使用有效指标不报错。"""
        cfg = EvalConfig(metrics=sorted(VALID_METRICS))
        assert len(cfg.metrics) == 5

    def test_empty_metrics_list_is_allowed(self) -> None:
        """空指标列表不报错（跳过所有指标计算）。"""
        cfg = EvalConfig(metrics=[])
        assert cfg.metrics == []

    def test_single_valid_metric_is_allowed(self) -> None:
        """单个有效指标不报错。"""
        for metric in VALID_METRICS:
            cfg = EvalConfig(metrics=[metric])
            assert cfg.metrics == [metric]
