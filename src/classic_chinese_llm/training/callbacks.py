"""训练回调系统 —— 插件式钩子, 用于日志记录、checkpoint 保存和早停。"""

from __future__ import annotations

import time
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from classic_chinese_llm.utils.logging_config import get_logger

if TYPE_CHECKING:
    from classic_chinese_llm.training.trainer import Trainer

logger = get_logger(__name__)


class Callback(ABC):
    """回调基类。所有钩子方法默认为 no-op, 子类选择性覆写。"""

    def on_train_begin(self, trainer: Trainer) -> None:
        """训练开始前调用。"""

    def on_step_end(self, trainer: Trainer, loss: float, lr: float) -> None:
        """每个 optimizer step 之后调用。"""

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        """每次评估结束后调用。"""

    def on_epoch_end(self, trainer: Trainer) -> None:
        """每个 epoch 结束时调用。"""

    def on_train_end(self, trainer: Trainer) -> None:
        """训练完成 (正常终止或 early stop) 时调用。"""


class LoggingCallback(Callback):
    """训练日志回调。

    记录并显示:
    - step / loss / lr
    - tokens/sec (吞吐量)
    - GPU 显存使用量
    - best_loss

    输出: 终端 tqdm 进度条 + 文件日志。
    """

    def __init__(self, log_dir: Path, log_every: int = 10) -> None:
        self.log_dir = Path(log_dir)
        self.log_every = log_every
        self._losses: list[float] = []
        self._start_time: float = 0.0
        self._pbar: tqdm | None = None

    def on_train_begin(self, trainer: Trainer) -> None:
        self._start_time = time.time()
        total = trainer.total_steps
        self._pbar = tqdm(
            total=total,
            initial=trainer.global_step,
            desc="Training",
            dynamic_ncols=True,
        )

    def on_step_end(self, trainer: Trainer, loss: float, lr: float) -> None:
        self._losses.append(loss)

        if trainer.global_step % self.log_every == 0 and self._pbar is not None:
            elapsed = time.time() - self._start_time
            batch_size = trainer.config.training.batch_size
            accum = trainer.config.training.gradient_accumulation_steps
            tokens = batch_size * trainer.config.model.max_seq_len * accum * len(self._losses)
            tps = tokens / max(elapsed, 0.001)

            gpu_mem = (
                torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            )
            avg_loss = sum(self._losses) / max(len(self._losses), 1)

            self._pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{tps:.0f}",
                    "mem": f"{gpu_mem:.1f}GB",
                    "best": f"{trainer.best_loss:.4f}",
                }
            )
            self._pbar.update(len(self._losses))

            self._losses = []
            self._start_time = time.time()

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        val_loss = metrics.get("val_loss", float("nan"))
        logger.info(
            "Eval @ step %d: val_loss=%.4f, best_loss=%.4f",
            trainer.global_step,
            val_loss,
            trainer.best_loss,
        )

    def on_train_end(self, trainer: Trainer) -> None:
        if self._pbar is not None:
            self._pbar.close()
        logger.info(
            "训练完成: step=%d, best_loss=%.4f",
            trainer.global_step,
            trainer.best_loss,
        )


class CheckpointCallback(Callback):
    """Checkpoint 保存回调。

    在 Trainer 中已集成了周期性保存和 best 保存逻辑,
    此回调主要负责额外的日志记录。
    """

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < trainer.best_loss - 1e-6:
            logger.info("新的 best loss: %.6f @ step %d", val_loss, trainer.global_step)


class EarlyStoppingCallback(Callback):
    """早停回调。

    如果连续 patience 次评估后 val_loss 没有改善 (下降超过 min_delta),
    则设置 trainer._should_stop = True 终止训练。

    Args:
        patience: 容忍的评估次数。
        min_delta: 视为"改善"的最小 loss 下降量。
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best_loss = float("inf")

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logger.info(
                    "早停触发: val_loss 连续 %d 次未改善 (best=%.4f, current=%.4f)",
                    self.patience,
                    self._best_loss,
                    val_loss,
                )
                trainer._should_stop = True
