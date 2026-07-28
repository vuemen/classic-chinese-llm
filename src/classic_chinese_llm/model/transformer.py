"""Transformer 模型组装 —— TransformerBlock 与完整 TransformerLM。

将 layers.py 中的基础组件组装为完整的 Decoder-only Transformer 语言模型。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from classic_chinese_llm.config.settings import ModelConfig
from classic_chinese_llm.model.layers import (
    MultiHeadAttention,
    RMSNorm,
    SwiGLUFFN,
    precompute_freqs_cis,
)

# ═══════════════════════════════════════════════════════════════════════════
# TransformerBlock
# ═══════════════════════════════════════════════════════════════════════════


class TransformerBlock(nn.Module):
    """单个 Decoder-only Transformer 层 (Pre-norm 残差结构)。

    架构:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLUFFN(RMSNorm(x))

    Pre-norm 的优势:
    - 归一化在残差分支的输入端, 而非输出端
    - 残差路径的梯度可以"无障碍"地从深层传播到浅层
    - 训练更稳定, 对学习率 warmup 更宽容

    Args:
        d_model: 隐藏维度 (768)。
        n_heads: 注意力头数 (12)。
        d_ff: FFN 中间维度 (3072)。
        dropout: attention dropout 概率 (预训练阶段为 0.0)。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pre-norm 前向传播。

        Args:
            x: (batch, seq_len, d_model)。
            freqs_cis: RoPE 预计算频率。
            attention_mask: 可选的 padding mask。

        Returns:
            (batch, seq_len, d_model)。
        """
        # 残差连接: Pre-norm → Attention → 加法
        x = x + self.attn(self.attn_norm(x), freqs_cis, attention_mask)
        # 残差连接: Pre-norm → FFN → 加法
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# TransformerLM
# ═══════════════════════════════════════════════════════════════════════════


class TransformerLM(nn.Module):
    """完整的 Decoder-only Transformer 语言模型。

    架构:
        Input (token IDs)
          │
          ▼
        Token Embedding ─────────── Tied Weights ──────────────┐
          │                                                      │
          ▼                                                      │
        TransformerBlock × N  (N = n_layers)                    │
          │                                                      │
          ▼                                                      │
        Final RMSNorm                                            │
          │                                                      │
          ▼                                                      │
        LM Head ──────────── 权重共享 ───────────────────────────┘
          │
          ▼
        Logits (batch, seq_len, vocab_size)

    关键设计:
    - Tied embeddings: token_embedding.weight 与 lm_head.weight 共享
    - RoPE 频率预计算 + 注册为 buffer (不参与 state_dict, 确定性可重算)
    - LLaMA 风格的权重初始化 (小正态分布, 使初始残差分支输出接近零)

    Args:
        config: ModelConfig 配置对象。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer Blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final RMSNorm
        self.final_norm = RMSNorm(config.d_model)

        # LM Head (权重与 token_embedding 共享 / tied weights)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # RoPE 频率预计算 (所有层共享, persistent=False 不参与 state_dict)
        freqs_cis = precompute_freqs_cis(
            d_model=config.d_model // config.n_heads,  # head_dim
            max_seq_len=config.max_seq_len,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """LLaMA 风格的权重初始化。

        初始化策略:
        - nn.Linear: 小正态分布, std = 0.02 / sqrt(2 * n_layers)
          参考 DeepNet: 使残差分支的初始输出接近零
        - nn.Embedding: 正态分布, std = d_model^(-0.5)

        注意: lm_head 的权重与 token_embedding 共享,
        跳过 lm_head 初始化以避免覆盖 Embedding 的初始化结果。
        """
        if module is self.lm_head:
            return  # 权重与 token_embedding 共享, 已由 Embedding 分支初始化
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.d_model**-0.5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            input_ids: (batch, seq_len) 的 token ID 序列。
            attention_mask: 可选的 (batch, seq_len) padding mask, True=参与。

        Returns:
            (batch, seq_len, vocab_size) 的 logits。
        """
        # Token Embedding
        x = self.token_embedding(input_ids)

        # 逐层 TransformerBlock
        for layer in self.layers:
            x = layer(x, self.freqs_cis, attention_mask)

        # Final RMSNorm → LM Head
        x = self.final_norm(x)
        return self.lm_head(x)

    def get_num_params(self) -> int:
        """返回可训练参数总数。"""
        return sum(p.numel() for p in self.parameters())

    def get_device(self) -> torch.device:
        """返回模型所在设备。"""
        return next(self.parameters()).device
