"""Checkpoint 管理 —— 模型权重 + 优化器状态 + 训练元信息的保存与恢复。

特性:
- 原子写入: 临时文件 + rename 确保不损坏已有 checkpoint
- 自动轮换: 保留最近 N 个 step checkpoint + best checkpoint
- 完整恢复: 权重 / optimizer / RNG 状态 / 训练步数
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


# ─── 数据结构 ───────────────────────────────────────────────────────────


@dataclass
class CheckpointState:
    """Checkpoint 包含的完整训练状态。"""

    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any] | None
    global_step: int
    epoch: int = 0
    best_loss: float = float("inf")
    rng_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── 保存 ────────────────────────────────────────────────────────────────


def save_checkpoint(
    state: CheckpointState,
    checkpoint_dir: str | Path,
    *,
    tag: str = "step",
    max_checkpoints: int = 5,
) -> Path:
    """原子保存 checkpoint。

    流程:
    1. 写入临时文件 (.tmp)
    2. 原子 rename 为最终文件
    3. 清理超过 max_checkpoints 的旧 step checkpoint

    Args:
        state: CheckpointState 训练状态
        checkpoint_dir: checkpoint 目录
        tag: 文件标签，如 "step_1000" 或 "best"
        max_checkpoints: 保留最近 N 个 step checkpoint

    Returns:
        保存的 .pt 文件路径
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = ckpt_dir / f"checkpoint_{tag}.pt"
    meta_file = ckpt_dir / f"checkpoint_{tag}.json"

    # 构建保存对象
    checkpoint: dict[str, Any] = {
        "model_state_dict": state.model_state_dict,
        "optimizer_state_dict": state.optimizer_state_dict,
        "global_step": state.global_step,
        "epoch": state.epoch,
        "best_loss": state.best_loss,
        "rng_state": state.rng_state,
    }

    tmp_ckpt = ckpt_dir / f".checkpoint_{tag}.tmp"
    tmp_meta = ckpt_dir / f".checkpoint_{tag}_meta.tmp"

    try:
        torch.save(checkpoint, tmp_ckpt)
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(state.metadata, f, ensure_ascii=False, indent=2)

        # 原子重命名
        tmp_ckpt.replace(ckpt_file)
        tmp_meta.replace(meta_file)

    finally:
        tmp_ckpt.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)

    logger.info("Checkpoint 已保存: %s (step=%d)", ckpt_file, state.global_step)

    _rotate_checkpoints(checkpoint_dir, keep=max_checkpoints)

    return ckpt_file


# ─── 加载 ────────────────────────────────────────────────────────────────


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cuda",
) -> CheckpointState:
    """加载 checkpoint 并返回 CheckpointState。

    Args:
        checkpoint_path: .pt 文件路径
        map_location: torch.load 的设备映射

    Returns:
        CheckpointState（optimizer_state_dict 可能为 None）

    Raises:
        FileNotFoundError: checkpoint 文件不存在
        KeyError: checkpoint 缺少必要字段
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 文件不存在: {ckpt_path}")

    logger.info("正在加载 checkpoint: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    meta_path = ckpt_path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    state = CheckpointState(
        model_state_dict=checkpoint["model_state_dict"],
        optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
        global_step=checkpoint["global_step"],
        epoch=checkpoint.get("epoch", 0),
        best_loss=checkpoint.get("best_loss", float("inf")),
        rng_state=checkpoint.get("rng_state"),
        metadata=metadata,
    )

    logger.info(
        "Checkpoint 加载完成 | step=%d epoch=%d best_loss=%.4f",
        state.global_step,
        state.epoch,
        state.best_loss,
    )
    return state


# ─── 查找 ────────────────────────────────────────────────────────────────


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """在目录中查找最新的 checkpoint。

    查找顺序:
    1. checkpoint_latest.pt
    2. 按修改时间最新的 checkpoint_step_*.pt
    3. checkpoint_best.pt

    Returns:
        最新 checkpoint 路径，无文件返回 None
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    latest = ckpt_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest

    step_checkpoints = sorted(
        ckpt_dir.glob("checkpoint_step_*.pt"),
        key=_extract_step_number,
        reverse=True,
    )
    if step_checkpoints:
        return step_checkpoints[0]

    best = ckpt_dir / "checkpoint_best.pt"
    if best.exists():
        return best

    return None


def find_best_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """查找 best checkpoint。"""
    best = Path(checkpoint_dir) / "checkpoint_best.pt"
    return best if best.exists() else None


# ─── 内部函数 ────────────────────────────────────────────────────────────


def _extract_step_number(path: Path) -> int:
    """从 checkpoint_step_XXXX.pt 文件名中提取步数。"""
    name = path.stem  # e.g. "checkpoint_step_1000"
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def _rotate_checkpoints(checkpoint_dir: str | Path, keep: int) -> None:
    """清理旧 step checkpoint，保留最近 N 个。

    checkpoint_best.pt 和 checkpoint_latest.pt 不会被删除。
    """
    ckpt_dir = Path(checkpoint_dir)
    step_checkpoints = sorted(
        ckpt_dir.glob("checkpoint_step_*.pt"),
        key=_extract_step_number,
    )
    for old in step_checkpoints[: -max(0, keep)]:
        logger.debug("清理旧 checkpoint: %s", old)
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)
