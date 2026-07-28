#!/usr/bin/env python3
"""评测 CLI 入口 —— 模型评测与报告生成。

用法:
    # 基础评测（base 模型，无 chat template）
    python scripts/evaluate.py \\
        --checkpoint models/checkpoints/checkpoint_best.pt \\
        --test-data data/processed/eval/test.jsonl

    # 指令模型评测（使用 ChatML template）
    python scripts/evaluate.py \\
        --checkpoint models/checkpoints/sft_best.pt \\
        --test-data data/processed/eval/test.jsonl \\
        --chat-template classical_chinese_v1

    # 完整评测 + 报告输出
    python scripts/evaluate.py \\
        --checkpoint models/checkpoints/sft_best.pt \\
        --test-data data/processed/eval/test.jsonl \\
        --chat-template classical_chinese_v1 \\
        --output-dir reports/eval \\
        --max-samples 200

工作流程:
    1. 加载 YAML 配置（可选）
    2. 加载 tokenizer + checkpoint → 初始化模型
    3. 创建 Evaluator 实例
    4. 加载测试数据 → 逐条生成 → 计算指标
    5. 输出 JSON + Markdown 报告
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from classic_chinese_llm.config.paths import PathConfig  # noqa: E402
from classic_chinese_llm.evaluation.config import EvalConfig  # noqa: E402
from classic_chinese_llm.evaluation.evaluator import Evaluator  # noqa: E402
from classic_chinese_llm.model.generation import (  # noqa: E402
    GenerationConfig,
    Generator,
)
from classic_chinese_llm.model.transformer import TransformerLM  # noqa: E402
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer  # noqa: E402
from classic_chinese_llm.utils.checkpoint import load_checkpoint  # noqa: E402
from classic_chinese_llm.utils.device import detect_device  # noqa: E402
from classic_chinese_llm.utils.logging_config import (  # noqa: E402
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="文言文 LLM 评测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="模型 checkpoint 路径 (.pt 文件)",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="评测数据 JSONL 文件路径",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="SentencePiece 模型路径（默认自动查找 models/tokenizer/classical_chinese.model）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="报告输出目录（默认仅输出到终端）",
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help="ChatML 模板名称（指令模型评测时启用，如 classical_chinese_v1）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="评测样本上限（默认 500）",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="启用的指标（默认全部）。可选: perplexity bleu rouge_l char_accuracy classical_chinese_score",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度（默认 0 = 确定性）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="最大生成长度（默认 256）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. 初始化路径
    PathConfig.initialize(_PROJECT_ROOT)
    paths = PathConfig.get()

    # 2. 初始化日志
    setup_logging(level="INFO")
    logger.info("=== 文言文 LLM 评测 ===")

    # 3. 检测设备
    device = detect_device()
    logger.info("设备: %s", device)

    # 4. 加载 tokenizer
    tokenizer_path = args.tokenizer or str(paths.tokenizer_dir / "classical_chinese.model")
    if not Path(tokenizer_path).exists():
        logger.error("Tokenizer 未找到: %s", tokenizer_path)
        sys.exit(1)
    tokenizer = build_tokenizer(tokenizer_path)
    logger.info("Tokenizer 加载完成 (vocab_size=%d)", tokenizer.vocab_size)

    # 5. 加载模型
    logger.info("加载 checkpoint: %s", args.checkpoint)
    from classic_chinese_llm.config.settings import ModelConfig  # noqa: E402

    model_cfg = ModelConfig()
    model = TransformerLM(model_cfg)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)
    model.eval()
    logger.info("模型加载完成 (%s)", model.__class__.__name__)

    # 6. 创建 Generator
    gen_cfg = GenerationConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
    )
    generator = Generator(model, gen_cfg)

    # 7. 创建 Evaluator
    metrics = args.metrics or [
        "perplexity",
        "bleu",
        "rouge_l",
        "char_accuracy",
        "classical_chinese_score",
    ]
    eval_config = EvalConfig(
        max_samples=args.max_samples,
        metrics=metrics,
        generation=gen_cfg,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        chat_template=args.chat_template,
        checkpoint_name=Path(args.checkpoint).name,
        dataset_name=Path(args.test_data).name,
    )

    evaluator = Evaluator(
        model=model,
        generator=generator,
        tokenizer_encode_fn=tokenizer.encode,
        tokenizer_decode_fn=tokenizer.decode,
        config=eval_config,
    )

    # 8. 执行评测
    report = evaluator.evaluate(Path(args.test_data))

    # 9. 输出结果摘要
    logger.info("评测完成。指标:")
    for name, value in report.aggregate_metrics.items():
        logger.info("  %s: %.4f", name, value)


if __name__ == "__main__":
    main()
