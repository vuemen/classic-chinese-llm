"""TransformerBlock 与 TransformerLM 测试。"""

from __future__ import annotations

import pytest
import torch

from classic_chinese_llm.config.settings import ModelConfig
from classic_chinese_llm.model.layers import precompute_freqs_cis
from classic_chinese_llm.model.transformer import TransformerBlock, TransformerLM


def _make_model_config(**overrides: object) -> ModelConfig:
    """创建测试用的 ModelConfig（使用 Pydantic 允许的最小合法值）。"""
    defaults: dict[str, object] = {
        "vocab_size": 1000,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "d_ff": 256,
        "max_seq_len": 128,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# TransformerBlock
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransformerBlock:
    """TransformerBlock 单元测试。"""

    @pytest.fixture
    def block(self) -> TransformerBlock:
        return TransformerBlock(d_model=128, n_heads=4, d_ff=512)

    @pytest.fixture
    def freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(d_model=32, max_seq_len=64)  # head_dim = 128/4 = 32

    def test_output_shape(self, block: TransformerBlock, freqs_cis: torch.Tensor) -> None:
        """输出形状与输入一致。"""
        x = torch.randn(2, 32, 128)
        out = block(x, freqs_cis)
        assert out.shape == x.shape

    def test_residual_connection_exists(
        self, block: TransformerBlock, freqs_cis: torch.Tensor
    ) -> None:
        """残差连接存在: 输出 ≠ 子模块输出（含有原始输入）。"""
        x = torch.randn(1, 8, 128)
        # 直接计算 attention 输出
        attn_out = block.attn(block.attn_norm(x), freqs_cis)
        # block 的输出应该 = x + attn_out + FFN 残差（≠ 单独的 attn_out）
        full_out = block(x, freqs_cis)
        assert not torch.allclose(full_out, attn_out)

    def test_gradient_flows(self, block: TransformerBlock, freqs_cis: torch.Tensor) -> None:
        """所有子模块的梯度正常流动。"""
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = block(x, freqs_cis)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert block.attn.q_proj.weight.grad is not None
        assert block.ffn.gate_proj.weight.grad is not None

    def test_num_params(self, block: TransformerBlock) -> None:
        """每层参数量验证。"""
        n = sum(p.numel() for p in block.parameters())
        expected = (
            4 * 128 * 128  # Q/K/V/O: 4 × d_model²
            + 3 * 128 * 512  # SwiGLU: gate + up + down = 3 × d_model × d_ff
            + 128
            + 128  # RMSNorm × 2: 2 × d_model
        )
        assert n == expected

    def test_train_eval_mode(self, block: TransformerBlock, freqs_cis: torch.Tensor) -> None:
        """train/eval 模式切换正常。"""
        x = torch.randn(1, 8, 128)
        block.eval()
        out_eval = block(x, freqs_cis)
        block.train()
        out_train = block(x, freqs_cis)
        # 无 dropout 时 train/eval 输出应相同
        assert torch.equal(out_eval, out_train)


# ═══════════════════════════════════════════════════════════════════════════════
# TransformerLM
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransformerLM:
    """完整 TransformerLM 模型测试。"""

    @pytest.fixture
    def config(self) -> ModelConfig:
        return _make_model_config()

    @pytest.fixture
    def model(self, config: ModelConfig) -> TransformerLM:
        return TransformerLM(config)

    def test_forward_output_shape(self, model: TransformerLM) -> None:
        """forward 返回 (batch, seq_len, vocab_size) 的 logits。"""
        input_ids = torch.randint(0, 1000, (2, 32))
        logits = model(input_ids)
        assert logits.shape == (2, 32, 1000)

    def test_forward_single_token(self, model: TransformerLM) -> None:
        """单个 token 正常 forward。"""
        input_ids = torch.randint(0, 1000, (1, 1))
        logits = model(input_ids)
        assert logits.shape == (1, 1, 1000)

    def test_forward_max_seq_len(self, model: TransformerLM) -> None:
        """最大序列长度正常 forward。"""
        input_ids = torch.randint(0, 1000, (1, 64))
        logits = model(input_ids)
        assert logits.shape == (1, 64, 1000)

    def test_logits_not_nan(self, model: TransformerLM) -> None:
        """logits 不含 NaN 或 Inf。"""
        input_ids = torch.randint(0, 1000, (2, 32))
        logits = model(input_ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_get_num_params(self, model: TransformerLM, config: ModelConfig) -> None:
        """参数量与计算一致。"""
        total = model.get_num_params()
        # Embedding: vocab × d_model = 1000 × 128 = 128,000
        # Per layer: 4×128² + 3×128×512 + 2×128 = 65,536 + 196,608 + 256 = 262,400
        # 2 layers: 524,800
        # Final RMSNorm: 128
        # LM Head: tied with embedding = 0
        # Total: 128,000 + 524,800 + 128 = 652,928
        expected = (
            config.vocab_size * config.d_model
            + config.n_layers
            * (
                4 * config.d_model * config.d_model
                + 3 * config.d_model * config.d_ff
                + 2 * config.d_model
            )
            + config.d_model
        )
        assert total == expected

    def test_tied_weights(self, model: TransformerLM) -> None:
        """Embedding 和 LM Head 权重共享（同一块内存）。"""
        assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()

    def test_tied_weights_share_gradient(self, model: TransformerLM) -> None:
        """共享权重的梯度正确累加。"""
        input_ids = torch.randint(0, 1000, (1, 8))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        # embedding 和 lm_head 的 grad 应该指向同一块内存
        assert model.lm_head.weight.grad is not None
        assert model.token_embedding.weight.grad is not None
        assert model.lm_head.weight.grad.data_ptr() == model.token_embedding.weight.grad.data_ptr()

    def test_gradient_flows(self, model: TransformerLM) -> None:
        """所有层的梯度正常流动。"""
        input_ids = torch.randint(0, 1000, (1, 8))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} 梯度为 None"

    def test_freqs_cis_is_buffer(self, model: TransformerLM) -> None:
        """freqs_cis 注册为 buffer（不参与参数计数）。"""
        assert "freqs_cis" in dict(model.named_buffers())
        assert "freqs_cis" not in dict(model.named_parameters())

    def test_freqs_cis_not_in_state_dict(self, model: TransformerLM) -> None:
        """freqs_cis 不包含在 state_dict 中（persistent=False）。"""
        state_dict = model.state_dict()
        assert "freqs_cis" not in state_dict

    def test_get_device(self, model: TransformerLM) -> None:
        """get_device 返回正确的设备。"""
        device = model.get_device()
        assert isinstance(device, torch.device)

    def test_loss_decreases_on_overfit(self) -> None:
        """小批量过拟合测试: 在 5 个样本上 loss 应该下降。"""
        config = _make_model_config(vocab_size=1000, d_model=64, n_layers=1, n_heads=2, d_ff=256)
        model = TransformerLM(config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # 生成 5 个随机样本作为"过拟合目标"
        data = torch.randint(0, 1000, (5, 8))
        labels = (data[:, 1:]).clone()
        labels = torch.cat([labels, torch.randint(0, 1000, (5, 1))], dim=1)

        losses = []
        for _ in range(100):
            optimizer.zero_grad()
            logits = model(data)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 1000),
                labels[:, 1:].reshape(-1),
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "过拟合测试失败：loss 没有下降"

    def test_model_respects_config_params(self) -> None:
        """模型正确使用配置中的参数。"""
        config = _make_model_config(n_layers=6, n_heads=8)
        model = TransformerLM(config)
        assert len(model.layers) == 6
        assert model.config.n_layers == 6
        assert model.config.n_heads == 8

    def test_initialization_sets_reasonable_weights(self, model: TransformerLM) -> None:
        """权重初始化后值在合理范围内（不全是 0，不爆炸）。"""
        for name, param in model.named_parameters():
            if "weight" in name and param.ndim >= 2:
                std = param.std().item()
                assert 0.001 < std < 0.5, f"{name} std={std} 超出合理范围"
