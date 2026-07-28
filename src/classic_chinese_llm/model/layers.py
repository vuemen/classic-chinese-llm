"""Transformer 基础层组件 —— 纯 torch.nn 实现。

本模块包含 Decoder-only Transformer 的全部基础组件:
- RMSNorm: Root Mean Square Layer Normalization
- precompute_freqs_cis / apply_rotary_emb: RoPE 旋转位置编码
- MultiHeadAttention: 多头因果自注意力 (FlashAttention 后端)
- SwiGLUFFN: SwiGLU 门控前馈网络

所有组件均不依赖 HuggingFace 模型代码, 仅使用 torch.nn 原生模块。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════
# RMSNorm
# ═══════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    与 LayerNorm 的区别:
    - 不减去均值 (re-centering 不是归一化的核心)
    - 不添加平移参数 β (每层节省 d_model 个参数)
    - 仅保留缩放参数 γ (可学习的 gain)

    公式:
        RMSNorm(x) = x / RMS(x) · γ
        RMS(x) = sqrt(mean(x²) + ε)

    Args:
        d_model: 归一化维度。
        eps: 防止除零的小常数 (默认 1e-6)。
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入的最后维度做 RMS 归一化。

        Args:
            x: 任意形状的张量, 归一化在最后一维执行。

        Returns:
            与输入同形状的归一化张量。
        """
        # 在 float32 下计算以保证 BF16 下的数值精度
        x_float = x.float()
        rms = torch.sqrt((x_float**2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float / rms * self.weight).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# RoPE (Rotary Position Embedding)
# ═══════════════════════════════════════════════════════════════════════════


def precompute_freqs_cis(
    d_model: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """预计算 RoPE 旋转频率的 cos/sin 值。

    频率计算:
        θ_i = theta^(-2i/d_model),  i = 0, 1, ..., d_model/2 - 1
        旋转角度 = position × θ_i

    Args:
        d_model: 模型隐藏维度 (通常为 head_dim, 非全局 d_model)。
        max_seq_len: 最大序列长度。
        theta: RoPE 的 base frequency (默认 10000)。
        device: 目标设备。

    Returns:
        (max_seq_len, d_model//2, 2) 的 (cos, sin) 张量。
    """
    # θ_i: (d_model//2,)
    freq = 1.0 / (
        theta ** (torch.arange(0, d_model, 2, dtype=torch.float32, device=device) / d_model)
    )
    # positions: (max_seq_len,)
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    # angles: (max_seq_len, d_model//2)
    angles = torch.outer(positions, freq)
    # (max_seq_len, d_model//2, 2)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """将预计算的 RoPE 频率应用到 Q 或 K 张量。

    复数乘法公式:
        (a + bi) * (cos + sin i) = (a·cos - b·sin) + (a·sin + b·cos)i

    Args:
        x: (batch, n_heads, seq_len, head_dim) 的 Q 或 K 张量。
        freqs_cis: (seq_len, head_dim//2, 2) 的预计算 (cos, sin) 值。

    Returns:
        应用 RoPE 后的张量, 形状与输入相同。
    """
    seq_len = x.shape[2]
    freqs = freqs_cis[:seq_len].to(device=x.device)

    # 将 head_dim 拆为 (head_dim//2, 2), 每对相邻维度看作 (real, imag)
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)

    cos = freqs[..., 0].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    sin = freqs[..., 1].unsqueeze(0).unsqueeze(0)

    # 复数旋转
    x_out_real = x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin
    x_out_imag = x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos
    x_out = torch.stack([x_out_real, x_out_imag], dim=-1)

    return x_out.reshape_as(x).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# MultiHeadAttention
# ═══════════════════════════════════════════════════════════════════════════


class MultiHeadAttention(nn.Module):
    """多头因果自注意力。

    实现要点:
    1. Q/K/V 独立投影: 通过 nn.Linear 将 d_model 映射到 n_heads × head_dim
    2. RoPE 仅作用于 Q 和 K (V 不携带位置信息)
    3. 使用 F.scaled_dot_product_attention 后端 (自动启用 FlashAttention)
    4. is_causal=True 保证自回归 (每个 token 只能看到自身及之前的 token)

    Args:
        d_model: 隐藏维度 (768)。
        n_heads: 注意力头数 (12)。
        dropout: attention dropout 概率 (预训练阶段为 0.0)。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q/K/V/O 投影 (无 bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: (batch, seq_len, d_model) 的输入。
            freqs_cis: RoPE 预计算频率 (仅对 Q/K 应用)。
            attention_mask: 可选的 padding mask (True=屏蔽)。

        Returns:
            (batch, seq_len, d_model) 的输出。
        """
        batch, seq_len, _ = x.shape

        # 1. Q/K/V 投影 + 重塑为多头格式
        def _project_and_reshape(proj: nn.Linear) -> torch.Tensor:
            return proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = _project_and_reshape(self.q_proj)  # (B, n_heads, S, head_dim)
        k = _project_and_reshape(self.k_proj)
        v = _project_and_reshape(self.v_proj)

        # 2. 对 Q 和 K 应用 RoPE (V 不应用)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # 3. Scaled Dot-Product Attention
        # is_causal=True 等价于上三角 -inf mask, FlashAttention 内核有专门优化
        # 如果有额外的 padding mask, 需要合并 (此处简化为仅 causal mask)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attention_mask is None),
        )

        # 4. 合并多头 + O 投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(attn_output)


# ═══════════════════════════════════════════════════════════════════════════
# SwiGLU FFN
# ═══════════════════════════════════════════════════════════════════════════


class SwiGLUFFN(nn.Module):
    """SwiGLU 门控前馈网络。

    标准 FFN (ReLU):        x → W_up → ReLU → W_down
    SwiGLU FFN:             x → W_gate → SiLU ─┐
                            x → W_up ──────────→ ⊙ → W_down

    门控机制:
    - gate 分支: 学习"哪些信息可以通过" (SiLU 激活后取 (0, 1) 范围)
    - up 分支: 提供"信息内容"
    - 逐元素乘积实现选择性信息过滤

    Args:
        d_model: 输入/输出维度 (768)。
        d_ff: 中间层维度 (3072 = 4 × d_model)。
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (batch, seq_len, d_model)。

        Returns:
            (batch, seq_len, d_model)。
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
