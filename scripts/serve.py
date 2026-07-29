#!/usr/bin/env python3
"""API 服务 CLI 入口 —— 启动 FastAPI REST API（OpenAI 兼容）。

用法:
    # 启动 API 服务
    python scripts/serve.py --checkpoint models/checkpoints/sft_best.pt

    # 指定 host 和 port
    python scripts/serve.py \\
        --checkpoint models/checkpoints/sft_best.pt \\
        --host 0.0.0.0 \\
        --port 8000

    # 指定 tokenizer
    python scripts/serve.py \\
        --checkpoint models/checkpoints/sft_best.pt \\
        --tokenizer models/tokenizer/classical_chinese.model

API 端点:
    POST /v1/chat/completions  — 聊天补全（支持 SSE 流式）
    GET  /v1/models             — 可用模型列表
    GET  /health               — 健康检查

工作流程:
    1. 加载 tokenizer
    2. 加载 checkpoint → 创建 InferenceEngine
    3. 创建 FastAPI app → uvicorn 启动服务
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from classic_chinese_llm.chat.api import create_app  # noqa: E402
from classic_chinese_llm.config.paths import PathConfig  # noqa: E402
from classic_chinese_llm.config.settings import ModelConfig  # noqa: E402
from classic_chinese_llm.inference.engine import InferenceEngine  # noqa: E402
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer  # noqa: E402
from classic_chinese_llm.utils.logging_config import (  # noqa: E402
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="文言文 LLM API 服务",
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
        "--host",
        default="0.0.0.0",
        help="绑定地址（默认 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="绑定端口（默认 8000）",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载（开发模式，默认关闭）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. 初始化路径
    PathConfig.initialize(_PROJECT_ROOT)
    paths = PathConfig.get()

    # 2. 初始化日志
    setup_logging(level="INFO")
    logger.info("=== 文言文 LLM API 服务 ===")

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

    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=model_cfg,
        tokenizer_encode_fn=tokenizer.encode,
        tokenizer_decode_fn=tokenizer.decode,
    )
    logger.info("InferenceEngine 就绪")

    # 5. 创建 FastAPI 应用
    app = create_app(engine)

    # 6. 启动服务
    logger.info("启动 API 服务: http://%s:%d", args.host, args.port)
    logger.info("端点: POST /v1/chat/completions, GET /health")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
