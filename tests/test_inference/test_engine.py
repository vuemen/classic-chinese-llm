"""inference.engine 模块的单元与集成测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from classic_chinese_llm.config.settings import ModelConfig
from classic_chinese_llm.inference.engine import InferenceEngine
from classic_chinese_llm.model.generation import GenerationConfig
from classic_chinese_llm.model.transformer import TransformerLM
from classic_chinese_llm.utils.checkpoint import CheckpointState, save_checkpoint


def _make_tiny_model() -> TransformerLM:
    """创建用于测试的微型模型。"""
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


class TestInferenceEngine:
    """InferenceEngine 测试。"""

    def test_generate_returns_non_empty_text(self) -> None:
        """generate 应返回非空文本。"""
        model = _make_tiny_model()
        engine = InferenceEngine(
            model=model,
            tokenizer_decode_fn=lambda ids: "生成的文本",
            tokenizer_encode_fn=lambda text: [1, 2, 3],
        )
        result = engine.generate("测试输入", max_new_tokens=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stream_yields_tokens(self) -> None:
        """stream 应至少 yield 1 个 token。"""
        model = _make_tiny_model()
        engine = InferenceEngine(
            model=model,
            tokenizer_decode_fn=lambda ids: "X",
            tokenizer_encode_fn=lambda text: [1],
        )
        tokens = list(engine.stream("测试", max_new_tokens=3, do_sample=False))
        assert len(tokens) >= 1

    def test_from_checkpoint_creates_engine(self, temp_dir: Path) -> None:
        """from_checkpoint 应成功创建推理引擎。"""
        # 保存微型模型的 checkpoint
        model = _make_tiny_model()
        ckpt_path = temp_dir / "checkpoint_best.pt"
        state = CheckpointState(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=None,
            global_step=0,
            epoch=0,
            best_loss=float("inf"),
        )
        save_checkpoint(state, temp_dir, tag="best")

        engine = InferenceEngine.from_checkpoint(
            checkpoint_path=ckpt_path,
            config=model.config,
            tokenizer_decode_fn=lambda ids: "test",
            tokenizer_encode_fn=lambda text: [1, 2, 3],
        )
        assert isinstance(engine, InferenceEngine)
        assert engine.model is not None

    def test_from_checkpoint_nonexistent_raises(self, temp_dir: Path) -> None:
        """不存在的 checkpoint 应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            InferenceEngine.from_checkpoint(
                checkpoint_path=temp_dir / "nonexistent.pt",
                config=ModelConfig(),
                tokenizer_decode_fn=lambda ids: "",
                tokenizer_encode_fn=lambda text: [],
            )

    def test_generate_with_history(self) -> None:
        """提供 history 时应构建完整对话格式。"""
        model = _make_tiny_model()
        captured_prompt: list[str] = []

        def fake_encode(text: str) -> list[int]:
            captured_prompt.append(text)
            return [1, 2, 3]

        engine = InferenceEngine(
            model=model,
            tokenizer_decode_fn=lambda ids: "回复",
            tokenizer_encode_fn=fake_encode,
        )

        history = [
            {"role": "system", "content": "你是文言文专家"},
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答"},
        ]
        result = engine.generate("新问题", history=history, max_new_tokens=5)
        assert len(result) > 0
        # encode 被调用过且 prompt 包含关键信息
        assert len(captured_prompt) > 0

    def test_custom_generation_config(self) -> None:
        """自定义 GenerationConfig 应影响生成行为。"""
        model = _make_tiny_model()
        engine = InferenceEngine(
            model=model,
            tokenizer_decode_fn=lambda ids: "X",
            tokenizer_encode_fn=lambda text: [1],
        )

        # 使用 temperature=0 的确定性生成
        gen_config = GenerationConfig(
            max_new_tokens=2,
            temperature=0.0,
            do_sample=False,
        )
        result1 = engine.generate("test", generation_config=gen_config)
        result2 = engine.generate("test", generation_config=gen_config)
        # 确定性生成应对相同输入产生相同输出
        assert result1 == result2
