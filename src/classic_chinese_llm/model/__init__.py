"""模型层 —— Decoder-only Transformer 的纯 torch.nn 实现。

主要组件:
- RMSNorm: Root Mean Square Layer Normalization
- MultiHeadAttention: 多头因果自注意力 (FlashAttention 后端)
- SwiGLUFFN: SwiGLU 门控前馈网络
- TransformerBlock: 单个 Pre-norm Transformer 层
- TransformerLM: 完整 Decoder-only Transformer 语言模型
- Generator: 自回归文本生成器
"""

from classic_chinese_llm.model.generation import (
    GenerationConfig,
    Generator,
)
from classic_chinese_llm.model.layers import (
    MultiHeadAttention,
    RMSNorm,
    SwiGLUFFN,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from classic_chinese_llm.model.transformer import TransformerBlock, TransformerLM

__all__ = [
    "RMSNorm",
    "MultiHeadAttention",
    "SwiGLUFFN",
    "apply_rotary_emb",
    "precompute_freqs_cis",
    "TransformerBlock",
    "TransformerLM",
    "GenerationConfig",
    "Generator",
]
