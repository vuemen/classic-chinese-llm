"""评测器 —— 加载模型 → 批量评测 → 输出报告。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from classic_chinese_llm.evaluation.config import EvalConfig
from classic_chinese_llm.evaluation.metrics import (
    calc_bleu,
    calc_char_accuracy,
    calc_perplexity,
    calc_rouge_l,
)
from classic_chinese_llm.evaluation.report import EvalReport, EvalSample
from classic_chinese_llm.model.generation import Generator
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# 类型别名: tokenizer 解码函数
DecodeFn = Callable[[list[int]], str]


class Evaluator:
    """模型评测器。

    职责:
    1. 加载 ChatML/JSONL 格式的测试数据集
    2. 逐条生成模型预测
    3. 计算所有已注册指标
    4. 生成并输出评测报告

    Args:
        model: TransformerLM 模型实例 (置于目标设备)。
        generator: Generator 生成器实例。
        tokenizer_decode_fn: 将 token ID 列表解码为文本的函数。
        config: 评测配置。
    """

    def __init__(
        self,
        model: nn.Module,
        generator: Generator,
        tokenizer_decode_fn: DecodeFn,
        config: EvalConfig | None = None,
    ) -> None:
        self.model = model
        self.generator = generator
        self.decode = tokenizer_decode_fn
        self.config = config or EvalConfig()
        self._device = next(model.parameters()).device

    def evaluate(self, test_data_path: Path) -> EvalReport:
        """执行完整评测流程。

        Args:
            test_data_path: ChatML 格式的测试数据 JSONL 文件。

        Returns:
            EvalReport: 包含所有样本和聚合指标的评测报告。
        """
        logger.info("开始评测: %s", test_data_path)
        samples_data = self._load_samples(test_data_path)

        # 步骤 1: 批量生成回答（同时累积 perplexity 的 loss）
        eval_samples, ppl_loss, ppl_tokens = self._generate_responses(samples_data)

        # 步骤 2: 计算聚合指标
        aggregate = self._compute_metrics(eval_samples, ppl_loss, ppl_tokens)

        # 步骤 3: 构建报告
        report = EvalReport.create(
            config=self.config,
            samples=eval_samples,
            aggregate_metrics=aggregate,
            model_info=self._get_model_info(),
        )

        # 输出报告
        logger.info(report.summary())
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            json_path = self.config.output_dir / "eval_report.json"
            report.to_json(json_path)
            logger.info("评测报告已保存: %s", json_path)

        return report

    def _generate_responses(
        self, samples_data: list[dict[str, Any]]
    ) -> tuple[list[EvalSample], float, int]:
        """逐条生成模型回答并计算逐样本指标。

        对每条样本: 编码 prompt → 生成 → 解码 → 计算逐样本指标。
        同时累积 perplexity 计算所需的 cross-entropy loss。

        Args:
            samples_data: 加载后的样本字典列表。

        Returns:
            (eval_samples, total_loss, total_tokens):
            - eval_samples: 评测样本列表
            - total_loss: 累积 cross-entropy loss
            - total_tokens: 累积 token 数
        """
        eval_samples: list[EvalSample] = []
        total_loss = 0.0
        total_tokens = 0
        for data in samples_data:
            sample = self._generate_one(data)
            eval_samples.append(sample)

            # 累积 perplexity 计算所需的 loss
            if "perplexity" in self.config.metrics:
                ref_ids = data.get("reference_ids", [])
                pred_ids = data.get("prediction_ids", [])
                if ref_ids and pred_ids:
                    sample_loss = self._compute_cross_entropy(
                        torch.tensor([pred_ids], device=self._device),
                        torch.tensor([ref_ids], device=self._device),
                    )
                    total_loss += sample_loss * len(ref_ids)
                    total_tokens += len(ref_ids)

        return eval_samples, total_loss, total_tokens

    def _generate_one(self, data: dict[str, Any]) -> EvalSample:
        """生成单条样本。

        Args:
            data: 包含 prompt、reference 和 input_ids 的样本字典。

        Returns:
            EvalSample: 单条评测结果。
        """
        prompt = data["prompt"]
        reference = data["reference"]

        # 生成预测
        input_ids = torch.tensor([data["input_ids"]], device=self._device, dtype=torch.long)
        output_ids = self.generator.generate(input_ids, self.config.generation)

        # 解码: 仅取生成部分
        prompt_len = input_ids.size(1)
        new_token_ids = output_ids[0, prompt_len:].tolist()
        prediction = self.decode(new_token_ids)

        # 逐样本指标
        sample_metrics: dict[str, float] = {}
        if "bleu" in self.config.metrics:
            sample_metrics["bleu"] = calc_bleu([prediction], [[reference]])
        if "rouge_l" in self.config.metrics:
            sample_metrics["rouge_l"] = calc_rouge_l([prediction], [[reference]])
        if "char_accuracy" in self.config.metrics:
            sample_metrics["char_accuracy"] = calc_char_accuracy([prediction], [[reference]])

        return EvalSample(
            prompt=prompt,
            reference=reference,
            prediction=prediction,
            metrics=sample_metrics,
        )

    def _load_samples(self, path: Path) -> list[dict[str, Any]]:
        """从 JSONL 文件加载评测样本。

        支持两种格式:
        1. ChatML: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
        2. 简单格式: {"prompt": "...", "reference": "..."}

        Args:
            path: JSONL 文件路径。

        Returns:
            list[dict]: 每条包含 prompt, reference, input_ids, reference_ids。
        """
        samples: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                # 提取 prompt 和 reference
                extracted = self._extract_prompt_ref(record)
                if extracted is None:
                    continue

                samples.append(extracted)

                if len(samples) >= self.config.max_samples:
                    break

        logger.info("加载 %d 条评测样本", len(samples))
        return samples

    def _extract_prompt_ref(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """从单条记录提取 prompt、reference 和 token IDs。

        Args:
            record: 原始 JSONL 记录。

        Returns:
            dict 或 None (如果记录无效)。
        """
        messages = record.get("messages")
        if messages and isinstance(messages, list) and len(messages) >= 2:
            # ChatML 格式: 取 user 作为 prompt, assistant 作为 reference
            user_text = ""
            asst_text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    user_text = content
                elif role == "assistant":
                    asst_text = content

            if not user_text or not asst_text:
                return None

            # 将 user 文本编码为 input_ids (需要 tokenizer)
            # 这里记录文本形式，由调用方通过 tokenizer_decode_fn 处理
            return {
                "prompt": user_text,
                "reference": asst_text,
                "input_ids": [ord(ch) % 100 for ch in user_text],
                "reference_ids": [ord(ch) % 100 for ch in asst_text],
            }

        # 简单格式: prompt + reference
        prompt = record.get("prompt", "")
        reference = record.get("reference", "")
        if prompt and reference:
            return {
                "prompt": prompt,
                "reference": reference,
                "input_ids": [ord(ch) % 100 for ch in prompt],
                "reference_ids": [ord(ch) % 100 for ch in reference],
            }

        return None

    def _compute_metrics(
        self,
        samples: list[EvalSample],
        total_loss: float = 0.0,
        total_tokens: int = 0,
    ) -> dict[str, float]:
        """从样本列表计算 corpus-level 聚合指标。

        对各样本的指标取平均值。perplexity 使用累积的
        cross-entropy loss 计算。

        Args:
            samples: 评测样本列表。
            total_loss: 累积 cross-entropy loss（来自 _generate_responses）。
            total_tokens: 累积 token 数。

        Returns:
            dict[str, float]: 聚合后的指标值。
        """
        result: dict[str, float] = {}

        if "perplexity" in self.config.metrics and total_tokens > 0:
            avg_loss = total_loss / total_tokens
            result["perplexity"] = calc_perplexity(avg_loss)

        # 对各指标取样本平均值
        for metric_name in ["bleu", "rouge_l", "char_accuracy"]:
            if metric_name not in self.config.metrics:
                continue
            values = [s.metrics.get(metric_name) for s in samples]
            valid = [v for v in values if v is not None]
            if valid:
                result[metric_name] = sum(valid) / len(valid)

        return result

    def _compute_cross_entropy(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """计算 cross-entropy loss (用于 perplexity)。

        Args:
            input_ids: (1, seq_len) 输入 token IDs。
            labels: (1, seq_len) 目标 token IDs。

        Returns:
            float: 平均 loss 值。
        """
        with torch.no_grad():
            logits = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return float(loss.item())

    def _get_model_info(self) -> dict[str, Any]:
        """收集模型元信息。

        Returns:
            dict: 模型名称、参数量、设备等信息。
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "model_class": self.model.__class__.__name__,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self._device),
        }
