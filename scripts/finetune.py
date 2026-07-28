#!/usr/bin/env python3
"""指令微调 CLI 入口 —— SFT (Supervised Fine-Tuning).

用法:
    python scripts/finetune.py \
        --config configs/sft.yaml \
        --pretrained-checkpoint models/checkpoints/checkpoint_best.pt

工作流程:
    1. 加载 YAML 配置
    2. 加载 tokenizer
    3. 加载预训练 checkpoint → 初始化模型权重
    4. 构建 SFTDataset (ChatML 格式) → DataLoader
    5. 启动 Trainer (仅 assistant token 计算 loss)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from classic_chinese_llm.config import SFTConfig, load_config  # noqa: E402
from classic_chinese_llm.config.paths import PathConfig  # noqa: E402
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer  # noqa: E402
from classic_chinese_llm.training.sft import SFTRunner  # noqa: E402
from classic_chinese_llm.utils.logging_config import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="文言文 LLM 指令微调 (SFT)",
    )
    parser.add_argument(
        "--config",
        default="configs/sft.yaml",
        help="YAML 配置文件路径 (默认 configs/sft.yaml)",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        required=True,
        help="预训练 checkpoint 路径 (例如 models/checkpoints/checkpoint_best.pt)",
    )
    parser.add_argument(
        "--train-data",
        default=None,
        help="训练数据路径 (覆盖默认值)",
    )
    parser.add_argument(
        "--val-data",
        default=None,
        help="验证数据路径 (覆盖默认值)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. 初始化项目路径
    PathConfig.initialize(_PROJECT_ROOT)
    paths = PathConfig.get()

    # 2. 加载配置
    config = load_config(args.config, SFTConfig)

    # 3. 初始化日志
    setup_logging(
        level=config.logging.level,
        log_file=str(paths.logs_dir / "sft.log"),
    )

    # 4. 加载 tokenizer
    tokenizer_path = paths.tokenizer_dir / "classical_chinese.model"
    tokenizer = build_tokenizer(tokenizer_path)

    # 5. 确定数据路径
    train_data = args.train_data or str(paths.processed_data_dir / "instructions" / "train.jsonl")
    val_data = args.val_data or str(paths.processed_data_dir / "instructions" / "val.jsonl")

    # 6. 启动 SFT
    runner = SFTRunner(
        config=config,
        train_data_path=train_data,
        val_data_path=val_data,
        pretrained_checkpoint=args.pretrained_checkpoint,
        tokenizer=tokenizer,
    )
    runner.run()


if __name__ == "__main__":
    main()
