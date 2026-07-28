"""模型层基础组件测试 —— RMSNorm, RoPE, MultiHeadAttention, SwiGLUFFN。"""

from __future__ import annotations

import pytest
import torch

from classic_chinese_llm.model.layers import (
    MultiHeadAttention,
    RMSNorm,
    SwiGLUFFN,
    apply_rotary_emb,
    precompute_freqs_cis,
)

# ═══════════════════════════════════════════════════════════════════════════════
# RMSNorm
# ═══════════════════════════════════════════════════════════════════════════════


class TestRMSNorm:
    """RMSNorm 单元测试。"""

    @pytest.fixture
    def rms_norm(self) -> RMSNorm:
        return RMSNorm(d_model=768)

    def test_output_shape(self, rms_norm: RMSNorm) -> None:
        """输出形状与输入一致。"""
        x = torch.randn(2, 128, 768)
        out = rms_norm(x)
        assert out.shape == x.shape

    def test_output_dtype_matches_input(self, rms_norm: RMSNorm) -> None:
        """输出 dtype 与输入一致（BF16 下也正确）。"""
        x = torch.randn(2, 64, 768, dtype=torch.bfloat16)
        out = rms_norm(x)
        assert out.dtype == torch.bfloat16

    def test_normalizes_to_unit_rms(self, rms_norm: RMSNorm) -> None:
        """归一化后 RMS ≈ 1（因 γ 初始化为全 1）。"""
        x = torch.randn(4, 256, 768) * 5.0 + 2.0  # 非零均值、大方差
        out = rms_norm(x)
        rms = torch.sqrt((out**2).mean(dim=-1))
        # 允许小误差（eps + 浮点）
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_learnable_weight_exists(self, rms_norm: RMSNorm) -> None:
        """γ 参数可学习。"""
        assert rms_norm.weight.requires_grad
        assert rms_norm.weight.shape == (768,)

    def test_weight_affects_output(self) -> None:
        """修改 γ 值会影响输出。"""
        norm_a = RMSNorm(d_model=64)
        norm_b = RMSNorm(d_model=64)
        norm_a.weight.data.fill_(2.0)
        norm_b.weight.data.fill_(1.0)

        x = torch.ones(1, 10, 64)
        out_a = norm_a(x)
        out_b = norm_b(x)
        # γ=2 的输出应该是 γ=1 的 2 倍（x 是常量 1，归一化后 RMS=1/sqrt(64)）
        assert not torch.allclose(out_a, out_b)

    def test_eps_prevents_zero_division(self) -> None:
        """全零输入时不产生 NaN。"""
        norm = RMSNorm(d_model=64, eps=1e-8)
        x = torch.zeros(2, 10, 64)
        out = norm(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flows(self, rms_norm: RMSNorm) -> None:
        """梯度正常流动。"""
        x = torch.randn(2, 16, 768, requires_grad=True)
        out = rms_norm(x)
        loss = out.sum()
        loss.backward()
        assert rms_norm.weight.grad is not None
        assert x.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# RoPE
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoPE:
    """RoPE 旋转位置编码测试。"""

    @pytest.fixture
    def freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(d_model=64, max_seq_len=128, theta=10000.0)

    def test_freqs_cis_shape(self, freqs_cis: torch.Tensor) -> None:
        """预计算频率形状正确。"""
        assert freqs_cis.shape == (128, 32, 2)  # (max_seq_len, head_dim//2, 2)

    def test_apply_rotary_emb_preserves_shape(self, freqs_cis: torch.Tensor) -> None:
        """应用 RoPE 后形状不变。"""
        x = torch.randn(2, 12, 64, 64)  # (B, n_heads, seq_len, head_dim)
        out = apply_rotary_emb(x, freqs_cis)
        assert out.shape == x.shape

    def test_apply_rotary_emb_different_positions_differ(self, freqs_cis: torch.Tensor) -> None:
        """不同位置的旋转结果不同。"""
        x = torch.ones(1, 1, 4, 64)  # 位置 0-3 有相同内容
        out = apply_rotary_emb(x, freqs_cis)
        # 不同位置的结果应该不同（旋转了不同角度）
        assert not torch.allclose(out[:, :, 0], out[:, :, 1])

    def test_rotation_preserves_norm(self, freqs_cis: torch.Tensor) -> None:
        """旋转操作不改变向量范数。"""
        x = torch.randn(2, 4, 16, 64)
        original_norm = x.norm(dim=-1)
        out = apply_rotary_emb(x, freqs_cis)
        rotated_norm = out.norm(dim=-1)
        assert torch.allclose(original_norm, rotated_norm, atol=1e-4)

    def test_freqs_cis_is_deterministic(self) -> None:
        """固定 seed 下频率值可复现。"""
        torch.manual_seed(42)
        freqs_a = precompute_freqs_cis(d_model=64, max_seq_len=32, theta=100.0)
        torch.manual_seed(42)
        freqs_b = precompute_freqs_cis(d_model=64, max_seq_len=32, theta=100.0)
        assert torch.equal(freqs_a, freqs_b)

    def test_freqs_cis_cos_sin_identity(self, freqs_cis: torch.Tensor) -> None:
        """cos² + sin² ≈ 1。"""
        cos = freqs_cis[..., 0]
        sin = freqs_cis[..., 1]
        assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# MultiHeadAttention
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiHeadAttention:
    """多头注意力测试。"""

    @pytest.fixture
    def attn(self) -> MultiHeadAttention:
        return MultiHeadAttention(d_model=768, n_heads=12)

    @pytest.fixture
    def freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(d_model=64, max_seq_len=64)

    def test_output_shape(self, attn: MultiHeadAttention, freqs_cis: torch.Tensor) -> None:
        """输出形状与输入一致。"""
        x = torch.randn(2, 32, 768)
        out = attn(x, freqs_cis)
        assert out.shape == x.shape

    def test_causal_mask_prevents_future_attention(
        self, attn: MultiHeadAttention, freqs_cis: torch.Tensor
    ) -> None:
        """因果 mask 确保 token i 不关注 token j (j > i)。"""
        x = torch.randn(1, 8, 768)
        x.requires_grad = True
        out = attn(x, freqs_cis)

        # 如果因果 mask 正确，修改 token 7 不应影响 token 0 的输出
        out_original = out[:, 0].clone()

        # 测试方式: 修改最后一个 token 然后看第一个 token 是否不变
        # （因为 is_causal=True，token 0 看不到 token 7）
        x_modified = x.clone()
        x_modified[:, 7] = torch.randn(768) * 100
        out_modified = attn(x_modified, freqs_cis)
        assert torch.allclose(out_original, out_modified[:, 0], atol=1e-3)

    def test_gradient_flows(self, attn: MultiHeadAttention, freqs_cis: torch.Tensor) -> None:
        """梯度正常流动。"""
        x = torch.randn(2, 16, 768, requires_grad=True)
        out = attn(x, freqs_cis)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert attn.q_proj.weight.grad is not None

    def test_different_batch_sizes(self, attn: MultiHeadAttention, freqs_cis: torch.Tensor) -> None:
        """不同 batch size 正常工作。"""
        for bs in [1, 2, 4, 8]:
            x = torch.randn(bs, 16, 768)
            out = attn(x, freqs_cis)
            assert out.shape == (bs, 16, 768)

    def test_num_params(self, attn: MultiHeadAttention) -> None:
        """参数量验证: 4 × d_model² = 4 × 768² = 2,359,296。"""
        n = sum(p.numel() for p in attn.parameters())
        assert n == 4 * 768 * 768  # Q, K, V, O 各一个投影

    def test_d_model_not_divisible_raises(self) -> None:
        """d_model 不能被 n_heads 整除时抛出 ValueError。"""
        with pytest.raises(ValueError, match="整除"):
            MultiHeadAttention(d_model=100, n_heads=3)


# ═══════════════════════════════════════════════════════════════════════════════
# SwiGLUFFN
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwiGLUFFN:
    """SwiGLU FFN 测试。"""

    @pytest.fixture
    def ffn(self) -> SwiGLUFFN:
        return SwiGLUFFN(d_model=768, d_ff=3072)

    def test_output_shape(self, ffn: SwiGLUFFN) -> None:
        """输出形状与输入一致。"""
        x = torch.randn(2, 128, 768)
        out = ffn(x)
        assert out.shape == x.shape

    def test_gradient_flows(self, ffn: SwiGLUFFN) -> None:
        """梯度正常流动。"""
        x = torch.randn(2, 16, 768, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert ffn.gate_proj.weight.grad is not None
        assert ffn.up_proj.weight.grad is not None
        assert ffn.down_proj.weight.grad is not None

    def test_gate_zero_produces_zero_output(self, ffn: SwiGLUFFN) -> None:
        """gate = 0 时输出 ≈ 0（验证门控机制）。"""
        with torch.no_grad():
            ffn.gate_proj.weight.fill_(0.0)
            ffn.up_proj.weight.normal_(0, 1)
            ffn.down_proj.weight.normal_(0, 1)

        x = torch.randn(1, 4, 768)
        out = ffn(x)
        # gate 全零 → SiLU(0)=0 → 输出应为零
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_num_params(self, ffn: SwiGLUFFN) -> None:
        """参数量验证: 3 × d_model × d_ff = 3 × 768 × 3072 = 7,077,888。"""
        n = sum(p.numel() for p in ffn.parameters())
        assert n == 3 * 768 * 3072

    def test_deterministic_output(self, ffn: SwiGLUFFN) -> None:
        """相同输入 → 相同输出（确定性）。"""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 768)
        out_a = ffn(x)
        out_b = ffn(x)
        assert torch.equal(out_a, out_b)
