"""推理引擎 —— 模型加载 + 确定/流式生成 + 多轮对话支持。

将 model 模块 (TransformerLM + Generator) 和 tokenizer 模块
封装为面向文本的推理接口。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import torch

from classic_chinese_llm.config.settings import ModelConfig
from classic_chinese_llm.model.generation import GenerationConfig, Generator
from classic_chinese_llm.model.transformer import TransformerLM
from classic_chinese_llm.utils.checkpoint import load_checkpoint
from classic_chinese_llm.utils.device import detect_device
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# 类型别名
EncodeFn = Callable[[str], list[int]]
DecodeFn = Callable[[list[int]], str]


class InferenceEngine:
    """推理引擎 —— 高层文本生成封装。

    职责:
    1. 管理模型、tokenizer 编解码函数、设备
    2. 提供 generate() 和 stream() 接口
    3. 处理多轮对话 history

    Args:
        model: TransformerLM 模型实例（必须在目标设备上）。
        tokenizer_decode_fn: token ID 列表 → 文本的解码函数。
        tokenizer_encode_fn: 文本 → token ID 列表的编码函数。
        generation_config: 默认生成参数。
    """

    def __init__(
        self,
        model: TransformerLM,
        tokenizer_decode_fn: DecodeFn,
        tokenizer_encode_fn: EncodeFn,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.model = model
        self.decode = tokenizer_decode_fn
        self.encode = tokenizer_encode_fn
        self.generation_config = generation_config or GenerationConfig()
        self.generator = Generator(model)
        self._device = next(model.parameters()).device

    # ─── 公开 API ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs: object,
    ) -> str:
        """非流式文本生成。

        Args:
            prompt: 用户输入文本。
            history: 多轮对话历史 (ChatML messages 格式)。
            generation_config: 覆盖默认生成参数。
            **kwargs: 其他生成参数 (如 max_new_tokens, temperature 等)。

        Returns:
            str: 模型生成的完整文本。
        """
        input_ids = self._build_input(prompt, history)
        gen_cfg = self._merge_config(generation_config, **kwargs)

        output_ids = self.generator.generate(input_ids, gen_cfg)
        new_tokens = output_ids[0, input_ids.size(1) :].tolist()

        return self.decode(new_tokens)

    def stream(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs: object,
    ) -> Iterator[str]:
        """流式文本生成。

        每次 yield 一个新 token 的文本。

        Args:
            prompt: 用户输入文本。
            history: 多轮对话历史。
            generation_config: 覆盖默认生成参数。
            **kwargs: 其他生成参数。

        Yields:
            str: 单个 token 对应的文本。
        """
        input_ids = self._build_input(prompt, history)
        gen_cfg = self._merge_config(generation_config, **kwargs)

        for token_id in self.generator.generate_stream(input_ids, gen_cfg):
            yield self.decode([token_id])

    # ─── 工厂方法 ────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: ModelConfig,
        tokenizer_decode_fn: DecodeFn,
        tokenizer_encode_fn: EncodeFn,
        device: str | None = None,
    ) -> InferenceEngine:
        """从 checkpoint 加载模型并创建引擎。

        Args:
            checkpoint_path: checkpoint 文件路径。
            config: 模型架构配置。
            tokenizer_decode_fn: 解码函数。
            tokenizer_encode_fn: 编码函数。
            device: 目标设备 ("cuda", "cpu" 等)，None 为自动检测。

        Returns:
            InferenceEngine 实例。

        Raises:
            FileNotFoundError: checkpoint 文件不存在。
        """
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint 文件不存在: {ckpt}")

        # 设备检测
        if device is None:
            device_info = detect_device()
            target_device = str(device_info.device)
        else:
            target_device = device

        logger.info("从 checkpoint 加载模型: %s", ckpt)
        state = load_checkpoint(ckpt, map_location="cpu")

        model = TransformerLM(config)
        model.load_state_dict(state.model_state_dict, strict=False)
        model.to(target_device)
        model.eval()

        logger.info("模型已加载到 %s, 参数: %d", target_device, model.get_num_params())
        return cls(
            model=model,
            tokenizer_decode_fn=tokenizer_decode_fn,
            tokenizer_encode_fn=tokenizer_encode_fn,
        )

    # ─── 内部方法 ────────────────────────────────────────────────────────

    def _build_input(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> torch.Tensor:
        """构建模型输入 token IDs。

        如果提供 history，将其与当前 prompt 拼接为完整对话格式。

        Args:
            prompt: 当前用户输入。
            history: 多轮对话历史。

        Returns:
            (1, seq_len) 的 input_ids 张量。
        """
        if history:
            # 拼接历史 + 当前 prompt
            full_text_parts: list[str] = []
            for msg in history:
                full_text_parts.append(f"{msg['role']}: {msg['content']}")
            full_text_parts.append(f"user: {prompt}")
            full_text = "\n".join(full_text_parts)
            token_ids = self.encode(full_text)
        else:
            token_ids = self.encode(prompt)

        if not token_ids:
            token_ids = [0]  # fallback

        return torch.tensor([token_ids], device=self._device, dtype=torch.long)

    def _merge_config(
        self,
        generation_config: GenerationConfig | None,
        **kwargs: object,
    ) -> GenerationConfig:
        """合并默认生成配置与用户指定的参数。

        Args:
            generation_config: 用户自定义的完整配置（可选）。
            **kwargs: 单个生成参数覆盖。

        Returns:
            GenerationConfig: 合并后的配置。
        """
        if generation_config is not None:
            return generation_config

        defaults = self.generation_config

        def _get_int(key: str, fallback: int) -> int:
            val = kwargs.get(key)
            return int(val) if val is not None else fallback  # type: ignore[call-overload]

        def _get_float(key: str, fallback: float) -> float:
            val = kwargs.get(key)
            return float(val) if val is not None else fallback  # type: ignore[arg-type]

        def _get_bool(key: str, fallback: bool) -> bool:
            val = kwargs.get(key)
            return bool(val) if val is not None else fallback

        merged = GenerationConfig(
            max_new_tokens=_get_int("max_new_tokens", defaults.max_new_tokens),
            temperature=_get_float("temperature", defaults.temperature),
            top_k=_get_int("top_k", defaults.top_k),
            top_p=_get_float("top_p", defaults.top_p),
            repetition_penalty=_get_float("repetition_penalty", defaults.repetition_penalty),
            do_sample=_get_bool("do_sample", defaults.do_sample),
            eos_token_id=_get_int("eos_token_id", defaults.eos_token_id),
            pad_token_id=_get_int("pad_token_id", defaults.pad_token_id),
        )
        return merged
