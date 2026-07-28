"""文本生成器 —— 自回归生成 + 多种采样策略 + KV Cache 加速。

提供:
- GenerationConfig: 生成参数配置
- KVCache: 推理时的 Key/Value 缓存 (避免重复计算)
- Generator: 自回归生成器, 支持流式和非流式输出
"""

from __future__ import annotations

from collections.abc import Generator as PyGenerator
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from classic_chinese_llm.model.transformer import TransformerLM
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GenerationConfig
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationConfig:
    """生成参数配置。

    Attributes:
        max_new_tokens: 最大生成 token 数。
        temperature: 温度缩放 (1.0 = 无缩放, <1 = 更确定性, >1 = 更随机)。
        top_k: Top-K 采样 (0 = 不使用)。
        top_p: Top-P (Nucleus) 采样 (1.0 = 不使用)。
        repetition_penalty: 重复惩罚 (>1 惩罚重复, 1.0 = 不惩罚)。
        num_beams: Beam Search 的 beam 数 (1 = 贪心/采样模式)。
        do_sample: 是否使用采样 (False = 贪心 argmax)。
        eos_token_id: 结束 token ID。
        pad_token_id: 填充 token ID。
    """

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    eos_token_id: int = 3
    pad_token_id: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# KVCache
# ═══════════════════════════════════════════════════════════════════════════


class KVCache:
    """Key/Value 缓存 —— 逐 token 推理时避免重复计算历史 token 的 K/V。

    工作原理:
    - 每层缓存过去的 K 和 V 张量
    - 新 token 只计算增量 K/V, 然后追加到缓存
    - Attention 时 Q(新) 关注 K(缓存), 复杂度从 O(S²) 降为 O(S)

    Args:
        n_layers: Transformer 层数。
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """追加新的 K/V 并返回完整缓存。

        Args:
            layer_idx: 层索引 (0 ~ n_layers-1)。
            k: (B, n_heads, 1, head_dim) 新 token 的 Key。
            v: (B, n_heads, 1, head_dim) 新 token 的 Value。

        Returns:
            (full_k, full_v): (B, n_heads, seq_len_cache, head_dim)。
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
            return k, v

        existing_k: torch.Tensor = self.keys[layer_idx]  # type: ignore[assignment]
        existing_v: torch.Tensor = self.values[layer_idx]  # type: ignore[assignment]
        full_k = torch.cat([existing_k, k], dim=2)
        full_v = torch.cat([existing_v, v], dim=2)
        self.keys[layer_idx] = full_k
        self.values[layer_idx] = full_v
        return full_k, full_v

    def reset(self) -> None:
        """清空所有层的缓存 (开始新的生成时调用)。"""
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers


# ═══════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════


class Generator:
    """自回归文本生成器。

    支持:
    - Greedy: 每步选最高概率 token
    - Temperature Sampling: softmax 前将 logits 除以 temperature
    - Top-K: 仅保留概率最高的 K 个 token
    - Top-P (Nucleus): 保留累积概率 ≥ p 的最小 token 集合
    - Repetition Penalty: 对已生成的 token 施加惩罚
    - Beam Search: 维护 K 条候选序列 (num_beams > 1 时启用)

    Args:
        model: TransformerLM 模型实例 (必须在目标设备上)。
    """

    def __init__(self, model: TransformerLM) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
    ) -> torch.Tensor:
        """从输入的 token IDs 生成文本 (非流式)。

        Args:
            input_ids: (1, prompt_len) 或 (prompt_len,) 的 prompt token IDs。
            config: 生成参数配置。

        Returns:
            (1, prompt_len + new_tokens) 的完整序列 (含 prompt)。
        """
        if config is None:
            config = GenerationConfig()

        input_ids = self._ensure_batch_dim(input_ids)

        if config.num_beams > 1:
            return self._beam_search(input_ids, config)

        return self._sample_loop(input_ids, config)

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
    ) -> PyGenerator[int, None, None]:
        """从输入的 token IDs 逐 token 流式生成。

        每次 yield 一个新生成的 token ID。

        Args:
            input_ids: (1, prompt_len) 的 prompt token IDs。
            config: 生成参数配置。

        Yields:
            每一步新生成的 token ID (int)。
        """
        if config is None:
            config = GenerationConfig()

        input_ids = self._ensure_batch_dim(input_ids)
        generated = input_ids.clone()

        for _ in range(config.max_new_tokens):
            # 截断到 max_seq_len
            max_len = self.model.config.max_seq_len
            if generated.size(1) > max_len:
                generated = generated[:, -max_len:]

            logits = self.model(generated)[:, -1, :]  # (1, vocab_size)

            # 重复惩罚
            if config.repetition_penalty != 1.0:
                logits = _apply_repetition_penalty(logits, generated, config.repetition_penalty)

            # 温度
            if config.temperature > 0:
                logits = logits / config.temperature

            # Top-K
            if config.top_k > 0:
                logits = _top_k_filter(logits, config.top_k)

            # Top-P
            if config.top_p < 1.0:
                logits = _top_p_filter(logits, config.top_p)

            # 采样或贪心
            if config.do_sample and config.temperature > 0:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = int(next_token.item())

            # 检查 EOS
            if token_id == config.eos_token_id:
                break

            yield token_id
            generated = torch.cat([generated, next_token], dim=1)

    # ─── 内部方法 ────────────────────────────────────────────────────

    def _sample_loop(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """简单的自回归采样循环 (无 KV Cache 版本, 用于训练后的基本推理)。"""
        generated = input_ids.clone()
        max_len = self.model.config.max_seq_len

        for _ in range(config.max_new_tokens):
            if generated.size(1) > max_len:
                generated = generated[:, -max_len:]

            logits = self.model(generated)[:, -1, :]

            if config.repetition_penalty != 1.0:
                logits = _apply_repetition_penalty(logits, generated, config.repetition_penalty)

            if config.temperature > 0:
                logits = logits / config.temperature

            if config.top_k > 0:
                logits = _top_k_filter(logits, config.top_k)

            if config.top_p < 1.0:
                logits = _top_p_filter(logits, config.top_p)

            if config.do_sample and config.temperature > 0:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if next_token.item() == config.eos_token_id:
                break

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Beam Search 生成。

        维护 num_beams 条候选序列, 每次扩展时保留概率最高的 num_beams 条。
        """
        device = input_ids.device
        num_beams = config.num_beams
        max_len = self.model.config.max_seq_len

        # (1, prompt_len) → (num_beams, prompt_len) 扩展 beam
        prompt = input_ids.repeat(num_beams, 1)
        beam_scores = torch.zeros(num_beams, device=device)
        beam_scores[1:] = -1e9  # 仅第一条 beam 活跃

        done = torch.zeros(num_beams, dtype=torch.bool, device=device)
        generated = prompt.clone()

        for _ in range(config.max_new_tokens):
            if done.all():
                break

            if generated.size(1) > max_len:
                generated = generated[:, -max_len:]

            logits = self.model(generated)[:, -1, :]  # (num_beams, vocab_size)

            if config.temperature > 0:
                logits = logits / config.temperature

            # 对已完成的 beam, 将其所有候选概率设为零
            logits[done] = float("-inf")

            # 取 top (2 * num_beams) 个候选
            vocab_size = logits.size(-1)
            scores = F.log_softmax(logits, dim=-1)
            # beam_scores 广播: (num_beams, 1) + (num_beams, vocab_size) → (num_beams, vocab_size)
            next_scores = (beam_scores.unsqueeze(1) + scores).view(-1)
            top_scores, top_indices = torch.topk(next_scores, 2 * num_beams)

            # 解码 beam 索引和 token 索引
            beam_indices = top_indices // vocab_size  # 来源 beam
            token_indices = top_indices % vocab_size  # 新 token

            # 选择最优的 num_beams 条
            new_generated = []
            new_beam_scores = []
            new_done = []

            for i in range(num_beams):
                b_idx = beam_indices[i].item()
                t_idx = token_indices[i].item()
                score = top_scores[i]

                if done[b_idx]:
                    # 已完成的 beam 直接复制
                    new_generated.append(generated[b_idx].clone())
                    new_beam_scores.append(beam_scores[b_idx])
                    new_done.append(True)
                else:
                    seq = torch.cat([generated[b_idx], token_indices[i].unsqueeze(0)])
                    new_generated.append(seq)
                    new_beam_scores.append(score)
                    new_done.append(t_idx == config.eos_token_id)

            generated = torch.stack(new_generated)
            beam_scores = torch.tensor(new_beam_scores, device=device)
            done = torch.tensor(new_done, device=device)

        # 返回 score 最高的 beam
        best_idx = int(beam_scores.argmax().item())
        return generated[best_idx].unsqueeze(0)  # (1, seq_len)

    @staticmethod
    def _ensure_batch_dim(input_ids: torch.Tensor) -> torch.Tensor:
        """确保 input_ids 有 batch 维度。"""
        if input_ids.ndim == 1:
            return input_ids.unsqueeze(0)
        return input_ids


# ═══════════════════════════════════════════════════════════════════════════
# 采样辅助函数
# ═══════════════════════════════════════════════════════════════════════════


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """对已生成的 token 施加重复惩罚。

    Args:
        logits: (batch, vocab_size) 当前步的 logits。
        generated: (batch, seq_len) 已生成的 token 序列。
        penalty: 惩罚系数 (>1 惩罚重复)。

    Returns:
        惩罚后的 logits。
    """
    if penalty == 1.0:
        return logits

    for token_id in generated.unique():  # type: ignore[no-untyped-call]
        score = logits[:, token_id]
        logits[:, token_id] = torch.where(
            score > 0,
            score / penalty,
            score * penalty,
        )
    return logits


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-K 过滤: 仅保留概率最高的 K 个 token。

    Args:
        logits: (batch, vocab_size)。
        k: 保留的 token 数。

    Returns:
        过滤后的 logits (被排除的 token 设为 -inf)。
    """
    if k <= 0:
        return logits
    top_k_values, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold = top_k_values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Top-P (Nucleus) 过滤: 保留累积概率 ≥ p 的最小 token 集合。

    Args:
        logits: (batch, vocab_size)。
        p: 累积概率阈值 (0 < p ≤ 1)。

    Returns:
        过滤后的 logits (被排除的 token 设为 -inf)。
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 找到累积概率 > p 的位置并屏蔽
    mask = cumulative_probs > p
    mask[:, 1:] = mask[:, :-1].clone()  # 至少保留一个 token
    mask[:, 0] = False

    # 恢复原顺序
    unsorted_mask = torch.zeros_like(logits, dtype=torch.bool)
    unsorted_mask.scatter_(dim=-1, index=sorted_indices, src=mask)

    return logits.masked_fill(unsorted_mask, float("-inf"))
