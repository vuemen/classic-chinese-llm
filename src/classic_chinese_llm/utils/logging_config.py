"""日志系统初始化。

基于 stdlib logging + rich，提供:
- terminal: RichHandler 彩色输出 + traceback 美化
- file: FileHandler 纯文本记录
- 模块级 logger 通过 get_logger(__name__) 获取
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    *,
    rich_width: int = 120,
) -> None:
    """初始化全局日志配置。

    配置 root logger，所有模块通过 ``logging.getLogger(__name__)``
    自动继承此配置。

    Args:
        level: 日志级别 (DEBUG / INFO / WARNING / ERROR)
        log_file: 日志文件路径，None 表示仅输出到终端
        rich_width: Rich 控制台宽度
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))

    # 清除已有的 handler（避免重复添加）
    root.handlers.clear()

    # ── 终端 Handler (Rich) ──────────────────────────────────────
    console = Console(width=rich_width)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    rich_handler.setLevel(logging.DEBUG)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rich_handler)

    # ── 文件 Handler ──────────────────────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    # ── 抑制第三方库日志噪音 ───────────────────────────────────────
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。

    使用方式::

        from classic_chinese_llm.utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("训练开始")
    """
    return logging.getLogger(name)
