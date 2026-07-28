"""evaluation.evaluator 模块的单元与集成测试。"""

from __future__ import annotations

import json
from pathlib import Path

from classic_chinese_llm.config.settings import ModelConfig
from classic_chinese_llm.evaluation.config import EvalConfig
from classic_chinese_llm.evaluation.evaluator import Evaluator
from classic_chinese_llm.evaluation.report import EvalReport, EvalSample
from classic_chinese_llm.model.generation import Generator
from classic_chinese_llm.model.transformer import TransformerLM


def _make_tiny_model() -> TransformerLM:
    """创建用于测试的微型模型 (d_model=64, n_layers=2)。"""
    return TransformerLM(
        ModelConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
    )


def _write_test_jsonl(path: Path, samples: list[dict]) -> None:
    """写入测试用的 JSONL 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


class TestEvaluator:
    """Evaluator 测试。"""

    def test_evaluate_basic(self, temp_dir: Path) -> None:
        """使用微型模型完成基本评测流程。"""
        model = _make_tiny_model()
        generator = Generator(model)
        config = EvalConfig(
            max_samples=2,
            metrics=["perplexity", "char_accuracy"],
            output_dir=temp_dir,
        )

        # 构造测试数据: ChatML 格式
        test_data_path = temp_dir / "test_data.jsonl"
        _write_test_jsonl(
            test_data_path,
            [
                {
                    "messages": [
                        {"role": "user", "content": "天地"},
                        {"role": "assistant", "content": "玄黄"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "宇宙"},
                        {"role": "assistant", "content": "洪荒"},
                    ]
                },
            ],
        )

        evaluator = Evaluator(
            model=model,
            generator=generator,
            tokenizer_encode_fn=lambda text: [ord(ch) % 1000 for ch in text],
            tokenizer_decode_fn=lambda ids: "".join(chr(0x4E00 + (i % 100)) for i in ids),
            config=config,
        )

        report = evaluator.evaluate(test_data_path)
        assert isinstance(report, EvalReport)
        assert len(report.samples) == 2
        assert "char_accuracy" in report.aggregate_metrics

    def test_max_samples_truncation(self, temp_dir: Path) -> None:
        """max_samples 应正确截断评测样本数。"""
        model = _make_tiny_model()
        generator = Generator(model)
        config = EvalConfig(max_samples=1, output_dir=temp_dir)

        test_data_path = temp_dir / "test_data_2.jsonl"
        _write_test_jsonl(
            test_data_path,
            [
                {
                    "messages": [
                        {"role": "user", "content": "A"},
                        {"role": "assistant", "content": "B"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "C"},
                        {"role": "assistant", "content": "D"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "E"},
                        {"role": "assistant", "content": "F"},
                    ]
                },
            ],
        )

        evaluator = Evaluator(
            model=model,
            generator=generator,
            tokenizer_encode_fn=lambda text: [ord(ch) % 1000 for ch in text],
            tokenizer_decode_fn=lambda ids: str(ids),
            config=config,
        )

        report = evaluator.evaluate(test_data_path)
        # max_samples=1, 但至少 2 个样本（生成+参考对比需要）
        assert 1 <= len(report.samples) <= 1

    def test_report_saves_to_file(self, temp_dir: Path) -> None:
        """评测完成后的报告应保存到文件。"""
        model = _make_tiny_model()
        generator = Generator(model)
        output_dir = temp_dir / "eval_output"
        output_dir.mkdir()
        config = EvalConfig(max_samples=1, output_dir=output_dir)

        test_data_path = temp_dir / "test_data_3.jsonl"
        _write_test_jsonl(
            test_data_path,
            [
                {
                    "messages": [
                        {"role": "user", "content": "Q"},
                        {"role": "assistant", "content": "A"},
                    ]
                }
            ],
        )

        evaluator = Evaluator(
            model=model,
            generator=generator,
            tokenizer_encode_fn=lambda text: [ord(ch) % 1000 for ch in text],
            tokenizer_decode_fn=lambda ids: "decoded",
            config=config,
        )

        evaluator.evaluate(test_data_path)

        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) >= 1, f"期望至少 1 个 JSON 报告文件, 发现: {json_files}"

    def test_eval_sample_structure(self) -> None:
        """EvalSample 字段完整。"""
        sample = EvalSample(
            prompt="问天地",
            reference="答玄黄",
            prediction="答玄黄也",
            metrics={"char_accuracy": 0.6667},
        )
        result = sample.__dict__
        assert result["prompt"] == "问天地"
        assert result["reference"] == "答玄黄"
        assert result["prediction"] == "答玄黄也"
        assert "char_accuracy" in result["metrics"]
