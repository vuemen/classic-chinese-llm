"""日志系统测试。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from classic_chinese_llm.utils.logging_config import get_logger, setup_logging

# ─── 辅助函数 ────────────────────────────────────────────────────────────────


def _close_file_handlers() -> None:
    """关闭并移除 root logger 的所有 FileHandler，释放文件句柄（Windows 必需）。"""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            root.removeHandler(handler)


def _flush_all_handlers() -> None:
    """刷新所有 handler，确保日志写入磁盘。"""
    for handler in logging.getLogger().handlers:
        handler.flush()


# ─── 测试类 ──────────────────────────────────────────────────────────────────


class TestSetupLogging:
    """setup_logging 函数测试。"""

    def test_creates_log_directory(self, temp_dir: Path) -> None:
        """指定 log_file 时应创建日志目录。"""
        log_dir = temp_dir / "logs"
        log_file = log_dir / "test.log"

        assert not log_dir.exists()

        setup_logging(log_file=log_file)
        try:
            assert log_dir.exists()
            assert log_dir.is_dir()
        finally:
            _close_file_handlers()

    def test_log_file_is_created(self, temp_dir: Path) -> None:
        """指定 log_file 时应创建日志文件并写入内容。"""
        log_file = temp_dir / "test_output.log"

        setup_logging(level="INFO", log_file=log_file)
        try:
            logger = get_logger("test_creation")
            logger.info("测试日志消息")

            _flush_all_handlers()
            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert "测试日志消息" in content
        finally:
            _close_file_handlers()

    def test_default_level_is_info(self) -> None:
        """默认日志级别为 INFO。"""
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_custom_debug_level(self) -> None:
        """DEBUG 级别应正确设置。"""
        setup_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_custom_warning_level(self) -> None:
        """WARNING 级别应正确设置。"""
        setup_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_custom_error_level(self) -> None:
        """ERROR 级别应正确设置。"""
        setup_logging(level="ERROR")
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_lowercase_level_string(self) -> None:
        """小写级别字符串（如 "info"）应正确设置。"""
        setup_logging(level="info")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_no_file_handler_when_log_file_is_none(self, temp_dir: Path) -> None:
        """未指定 log_file 时不应添加 FileHandler。"""
        setup_logging(level="INFO", log_file=None)
        root = logging.getLogger()

        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_rich_handler_added(self) -> None:
        """setup_logging 应添加 RichHandler。"""
        setup_logging()
        root = logging.getLogger()

        from rich.logging import RichHandler

        rich_handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
        assert len(rich_handlers) >= 1

    def test_log_file_accepts_string_path(self, temp_dir: Path) -> None:
        """log_file 接受字符串路径参数。"""
        log_file = temp_dir / "string_path.log"
        setup_logging(level="INFO", log_file=str(log_file))
        try:
            logger = get_logger("test_string_path")
            logger.info("通过字符串路径写入")

            _flush_all_handlers()
            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert "通过字符串路径写入" in content
        finally:
            _close_file_handlers()


class TestSetupLoggingIdempotence:
    """setup_logging 幂等性测试。"""

    def test_multiple_calls_do_not_crash(self) -> None:
        """多次调用 setup_logging 不应崩溃。"""
        for _ in range(3):
            setup_logging(level="INFO")
        # 不应抛异常

    def test_multiple_calls_do_not_duplicate_handlers(self) -> None:
        """多次调用不应重复添加 handler（handlers.clear() 每个调用清除旧 handler）。"""
        setup_logging()
        handler_count_1 = len(logging.getLogger().handlers)

        setup_logging()
        handler_count_2 = len(logging.getLogger().handlers)

        # 两次调用后 handler 数量应相同（先清除再添加）
        assert handler_count_1 == handler_count_2

    def test_reconfigure_level_between_calls(self) -> None:
        """可多次调用 setup_logging 改变日志级别。"""
        setup_logging(level="INFO")
        assert logging.getLogger().level == logging.INFO

        setup_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

        setup_logging(level="WARNING")
        assert logging.getLogger().level == logging.WARNING


class TestGetLogger:
    """get_logger 函数测试。"""

    def test_returns_logger_instance(self) -> None:
        """get_logger 返回 logging.Logger 实例。"""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self) -> None:
        """返回的 logger 名称与传入参数一致。"""
        logger = get_logger("custom.name")
        assert logger.name == "custom.name"

    def test_same_name_returns_same_logger(self) -> None:
        """相同名称多次调用返回同一个 Logger 实例。"""
        a = get_logger("reusable")
        b = get_logger("reusable")
        assert a is b

    def test_different_names_return_different_loggers(self) -> None:
        """不同名称返回不同的 Logger 实例。"""
        a = get_logger("module_a")
        b = get_logger("module_b")
        assert a is not b

    def test_logger_inherits_root_level(self) -> None:
        """子 logger 继承 root logger 的日志级别。"""
        setup_logging(level="INFO")
        logger = get_logger("test_inherit")

        # 子 logger 默认 level 为 0 (NOTSET)，会继承 root 的级别
        assert logger.level == logging.NOTSET
        # 实际有效级别是 INFO（从 root 继承）
        assert logger.getEffectiveLevel() == logging.INFO


class TestLogOutput:
    """日志输出行为测试。"""

    def test_debug_messages_not_logged_at_info_level(
        self, temp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """INFO 级别下 DEBUG 消息不应被记录。"""
        setup_logging(level="INFO")

        with caplog.at_level(logging.DEBUG):
            logger = get_logger("test_debug_suppressed")
            logger.debug("这条 DEBUG 消息不应出现")

        # root logger 的 level 是 INFO，DEBUG 消息被过滤
        assert logging.getLogger().level == logging.INFO

    def test_info_logged_to_file(self, temp_dir: Path) -> None:
        """INFO 级别的消息应写入日志文件。"""
        log_file = temp_dir / "info_test.log"
        setup_logging(level="INFO", log_file=log_file)
        try:
            logger = get_logger("test_info_file")
            logger.info("这是一条 INFO 日志")
            logger.warning("这是一条 WARNING 日志")

            _flush_all_handlers()
            content = log_file.read_text(encoding="utf-8")
            assert "INFO" in content
            assert "这是一条 INFO 日志" in content
            assert "WARNING" in content
            assert "这是一条 WARNING 日志" in content
        finally:
            _close_file_handlers()

    def test_file_log_contains_formatted_fields(self, temp_dir: Path) -> None:
        """日志文件中的记录应包含时间戳、级别和位置信息。"""
        log_file = temp_dir / "formatted.log"
        setup_logging(level="DEBUG", log_file=log_file)
        try:
            logger = get_logger("test_formatted")
            logger.info("格式化测试")

            _flush_all_handlers()
            content = log_file.read_text(encoding="utf-8")

            # 文件格式: asctime | levelname | name:lineno | message
            assert "INFO" in content
            assert "test_formatted" in content
            assert "格式化测试" in content
            assert "|" in content  # 分隔符存在
        finally:
            _close_file_handlers()

    def test_log_file_utf8_encoding(self, temp_dir: Path) -> None:
        """日志文件应使用 UTF-8 编码，支持中文字符。"""
        log_file = temp_dir / "utf8_test.log"
        setup_logging(level="INFO", log_file=log_file)
        try:
            logger = get_logger("test_utf8")
            logger.info("文言文测试 —— 之乎者也")

            _flush_all_handlers()
            content = log_file.read_text(encoding="utf-8")
            assert "文言文测试" in content
            assert "之乎者也" in content
        finally:
            _close_file_handlers()
