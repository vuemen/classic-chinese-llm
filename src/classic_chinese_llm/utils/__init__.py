"""工具模块 —— 日志、设备检测、Checkpoint 管理。

提供:
- setup_logging / get_logger: 日志系统
- detect_device / DeviceInfo / supports_bf16 / get_dtype / log_device_info: 设备检测
- CheckpointState / save_checkpoint / load_checkpoint / find_latest_checkpoint / find_best_checkpoint: Checkpoint
"""

from classic_chinese_llm.utils.checkpoint import (
    CheckpointState,
    find_best_checkpoint,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from classic_chinese_llm.utils.device import (
    DeviceInfo,
    detect_device,
    get_dtype,
    log_device_info,
    supports_bf16,
)
from classic_chinese_llm.utils.logging_config import get_logger, setup_logging

__all__ = [
    # logging
    "setup_logging",
    "get_logger",
    # device
    "DeviceInfo",
    "detect_device",
    "supports_bf16",
    "get_dtype",
    "log_device_info",
    # checkpoint
    "CheckpointState",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "find_best_checkpoint",
]
