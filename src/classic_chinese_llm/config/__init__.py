"""配置管理模块。

提供:
- load_config: YAML 配置加载入口（支持继承 + 环境变量覆盖）
- Settings / PretrainConfig / SFTConfig: Pydantic 配置模型
- PathConfig: 路径管理单例
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from classic_chinese_llm.config.paths import PathConfig
from classic_chinese_llm.config.settings import (
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    PretrainConfig,
    SchedulerConfig,
    Settings,
    SFTConfig,
    TrainingConfig,
)


def load_config[T: Settings](
    config_path: str | Path,
    config_cls: type[T] = Settings,  # type: ignore[assignment]
) -> T:
    """加载 YAML 配置文件并返回强类型配置对象。

    流程:
    1. 解析 YAML，递归处理 extends 继承
    2. pydantic-settings 自动注入 CCLLM_* 环境变量
    3. Pydantic 校验后返回

    Args:
        config_path: YAML 文件路径（相对路径基于 configs_dir 解析）
        config_cls: 期望的配置类型

    Returns:
        校验后的配置对象

    Raises:
        pydantic.ValidationError: 配置校验失败
        FileNotFoundError: 配置文件不存在
    """
    raw = _load_yaml_with_inheritance(config_path)
    return config_cls(**raw)


def _load_yaml_with_inheritance(config_path: str | Path) -> dict[str, Any]:
    """递归加载 YAML 文件，支持 extends 继承链。

    extends 值为相对路径时，相对于当前 YAML 所在目录解析。
    子配置覆盖父配置的同名字段（深度合并）。
    """
    path = Path(config_path)
    if not path.is_absolute():
        try:
            path = PathConfig.get().configs_dir / path
        except RuntimeError:
            # PathConfig 未初始化时，尝试相对于当前工作目录
            path = path.resolve()

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] | None = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # 递归解析 extends
    extends = raw.pop("extends", None)
    if extends:
        extends_path = path.parent / extends
        parent = _load_yaml_with_inheritance(extends_path)
        merged = _deep_merge(parent, raw)
        return merged

    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个字典，override 覆盖 base 的同名键。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


__all__ = [
    "load_config",
    "PathConfig",
    "Settings",
    "PretrainConfig",
    "SFTConfig",
    "DataConfig",
    "LoggingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
]
