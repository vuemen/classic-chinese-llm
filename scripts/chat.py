#!/usr/bin/env python3
"""对话 CLI 入口 —— 启动 Gradio Web UI 或命令行对话。

用法:
    # Gradio Web UI（浏览器中打开 http://localhost:7860）
    python scripts/chat.py --checkpoint models/checkpoints/sft_best.pt

    # 命令行对话模式
    python scripts/chat.py --checkpoint models/checkpoints/sft_best.pt --mode cli

    # 指定 tokenizer 和端口
    python scripts/chat.py \\
        --checkpoint models/checkpoints/sft_best.pt \\
        --tokenizer models/tokenizer/classical_chinese.model \\
        --port 7860

工作流程:
    1. 加载 tokenizer
    2. 加载 checkpoint → 创建 InferenceEngine
    3. 启动 Gradio Web UI 或 CLI 对话循环
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from classic_chinese_llm.chat.app import create_ui  # noqa: E402
from classic_chinese_llm.config.paths import PathConfig  # noqa: E402
from classic_chinese_llm.config.settings import ModelConfig  # noqa: E402
from classic_chinese_llm.inference.engine import InferenceEngine  # noqa: E402
from classic_chinese_llm.model.generation import GenerationConfig  # noqa: E402
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer  # noqa: E402
from classic_chinese_llm.utils.logging_config import (  # noqa: E402
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="文言文 LLM 对话界面",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="模型 checkpoint 路径 (.pt 文件)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="SentencePiece 模型路径（默认自动查找 models/tokenizer/classical_chinese.model）",
    )
    parser.add_argument(
        "--mode",
        choices=["gradio", "cli"],
        default="gradio",
        help="对话模式: gradio (Web UI) 或 cli (命令行)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="绑定地址（默认 127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="绑定端口（默认 7860）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度（默认 0.7）",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-P 采样阈值（默认 0.9）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="最大生成长度（默认 512）",
    )
    return parser.parse_args(argv)


def _cli_loop(engine: InferenceEngine) -> None:
    """命令行交互式对话循环。"""
    print("\n=== 文言文 LLM 命令行对话 ===\n")
    print("输入 'quit' 或 'exit' 退出，输入 'clear' 清空对话历史\n")

    history: list[dict[str, str]] = []
    while True:
        try:
            prompt = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            print("再见！")
            break
        if prompt.lower() == "clear":
            history = []
            print("对话历史已清空。")
            continue

        print("模型: ", end="", flush=True)
        response = engine.generate(prompt, history=history)
        print(response)
        print()

        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. 初始化路径
    PathConfig.initialize(_PROJECT_ROOT)
    paths = PathConfig.get()

    # 2. 初始化日志
    setup_logging(level="INFO", log_file=str(paths.logs_dir / "chat.log"))
    logger.info("=== 文言文 LLM 对话 ===")

    # 3. 加载 tokenizer
    tokenizer_path = args.tokenizer or str(paths.tokenizer_dir / "classical_chinese.model")
    if not Path(tokenizer_path).exists():
        logger.error("Tokenizer 未找到: %s", tokenizer_path)
        sys.exit(1)
    tokenizer = build_tokenizer(tokenizer_path)
    logger.info("Tokenizer 加载完成 (vocab_size=%d)", tokenizer.vocab_size)

    # 4. 加载模型
    logger.info("加载 checkpoint: %s", args.checkpoint)
    model_cfg = ModelConfig()
    gen_cfg = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
    )

    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=model_cfg,
        tokenizer_encode_fn=tokenizer.encode,
        tokenizer_decode_fn=tokenizer.decode,
    )
    engine.generation_config = gen_cfg
    logger.info("InferenceEngine 就绪")

    # 5. 启动对话
    if args.mode == "cli":
        _cli_loop(engine)
    else:
        logger.info("启动 Gradio Web UI: http://%s:%d", args.host, args.port)
        ui = create_ui(engine)
        ui.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
