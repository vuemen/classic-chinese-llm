#!/usr/bin/env python3
"""预训练 CLI 入口 —— Causal LM pretraining on 文言文 corpus.

用法:
    # 从头开始预训练
    python scripts/pretrain.py --config configs/pretrain.yaml

    # 从 checkpoint 恢复
    python scripts/pretrain.py --config configs/pretrain.yaml --resume

    # 环境变量覆盖
    CCLLM_TRAINING__BATCH_SIZE=16 python scripts/pretrain.py --config configs/pretrain.yaml

工作流程:
    1. 加载 YAML 配置 (with extends 继承 + 环境变量覆盖)
    2. 初始化日志 + 项目路径
    3. 加载训练好的 SentencePiece Unigram tokenizer
    4. 构建 PretrainDataset → DataLoader
    5. 创建 TransformerLM 模型
    6. 启动 Trainer 训练循环
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 path (支持任意位置执行)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from classic_chinese_llm.config import PretrainConfig, load_config  # noqa: E402
from classic_chinese_llm.config.paths import PathConfig  # noqa: E402
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer  # noqa: E402
from classic_chinese_llm.training.pretrain import PretrainRunner  # noqa: E402
from classic_chinese_llm.utils.logging_config import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="文言文 LLM 预训练 (Causal LM Pretraining)",
    )
    parser.add_argument(
        "--config",
        default="configs/pretrain.yaml",
        help="YAML 配置文件路径 (默认 configs/pretrain.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从最新 checkpoint 恢复训练",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="训练数据路径 (覆盖 YAML 中的设置)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. 初始化项目路径
    PathConfig.initialize(_PROJECT_ROOT)
    paths = PathConfig.get()

    # 2. 加载配置
    config = load_config(args.config, PretrainConfig)

    # 3. 初始化日志
    setup_logging(
        level=config.logging.level,
        log_file=str(paths.logs_dir / "pretrain.log"),
    )

    # 4. 加载 tokenizer
    tokenizer_path = paths.tokenizer_dir / "classical_chinese.model"
    tokenizer = build_tokenizer(tokenizer_path)

    # 5. 确定数据路径
    data_path = args.data_path or str(paths.processed_data_dir / "deduplicated.jsonl")

    # 6. 启动预训练
    runner = PretrainRunner(
        config=config,
        data_path=data_path,
        tokenizer=tokenizer,
    )
    runner.run()


if __name__ == "__main__":
    main()
