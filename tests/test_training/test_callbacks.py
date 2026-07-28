"""回调系统测试。—— Callback, LoggingCallback, EarlyStoppingCallback。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from classic_chinese_llm.training.callbacks import (
    Callback,
    EarlyStoppingCallback,
    LoggingCallback,
)


class TestCallback:
    """基类 Callback 测试。"""

    def test_all_hooks_exist(self) -> None:
        """所有钩子方法存在。"""
        cb = Callback()
        hooks = [
            "on_train_begin",
            "on_step_end",
            "on_eval_end",
            "on_epoch_end",
            "on_train_end",
        ]
        for hook in hooks:
            assert hasattr(cb, hook), f"缺少钩子: {hook}"

    def test_hooks_are_noop(self) -> None:
        """钩子默认为 no-op（调用不报错）。"""
        cb = Callback()
        mock_trainer = MagicMock()
        # 所有钩子调用不应抛出异常
        cb.on_train_begin(mock_trainer)
        cb.on_step_end(mock_trainer, loss=1.0, lr=0.001)
        cb.on_eval_end(mock_trainer, metrics={"val_loss": 0.5})
        cb.on_epoch_end(mock_trainer)
        cb.on_train_end(mock_trainer)


class TestEarlyStoppingCallback:
    """EarlyStoppingCallback 测试。"""

    def test_stops_after_patience(self) -> None:
        """连续 patience 次评估无改善后触发停止。"""
        cb = EarlyStoppingCallback(patience=3, min_delta=0.0)
        mock_trainer = MagicMock()
        mock_trainer._should_stop = False

        # 连续 4 次评估无改善
        for _ in range(4):
            cb.on_eval_end(mock_trainer, metrics={"val_loss": 2.0})

        assert mock_trainer._should_stop is True

    def test_improvement_resets_counter(self) -> None:
        """一次改善就重置计数器。"""
        cb = EarlyStoppingCallback(patience=3, min_delta=0.0)
        mock_trainer = MagicMock()
        mock_trainer._should_stop = False

        # 2 次无改善
        cb.on_eval_end(mock_trainer, metrics={"val_loss": 2.0})
        cb.on_eval_end(mock_trainer, metrics={"val_loss": 2.0})
        # 1 次改善
        cb.on_eval_end(mock_trainer, metrics={"val_loss": 1.5})
        # 2 次无改善
        cb.on_eval_end(mock_trainer, metrics={"val_loss": 2.0})
        cb.on_eval_end(mock_trainer, metrics={"val_loss": 2.0})

        assert mock_trainer._should_stop is False  # 还未达到 patience

    def test_min_delta_ignores_small_improvements(self) -> None:
        """小于 min_delta 的改善不计入改善。"""
        cb = EarlyStoppingCallback(patience=2, min_delta=0.01)
        mock_trainer = MagicMock()
        mock_trainer._should_stop = False

        # 改善 0.005 < min_delta=0.01 → 不算改善
        cb.on_eval_end(mock_trainer, metrics={"val_loss": 2.0})
        cb.on_eval_end(mock_trainer, metrics={"val_loss": 1.996})  # 改善 0.004
        cb.on_eval_end(mock_trainer, metrics={"val_loss": 2.0})

        assert mock_trainer._should_stop is True  # 连续 3 次"无改善"


class TestLoggingCallback:
    """LoggingCallback 测试。"""

    def test_on_train_begin_creates_progress_bar(self, temp_dir: Path) -> None:
        """on_train_begin 初始化进度条。"""
        cb = LoggingCallback(log_dir=temp_dir, log_every=10)
        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        mock_trainer.total_steps = 1000

        cb.on_train_begin(mock_trainer)
        assert cb._pbar is not None

        cb.on_train_end(mock_trainer)
