"""Pydantic 配置模型 — 类型安全的配置系统。

模块层级:
- Settings (基类): 通用参数 (seed, dtype, logging)
- PretrainConfig(Settings): 预训练配置
- SFTConfig(Settings): 指令微调配置
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── 叶子配置模型 ───────────────────────────────────────────────────────


class LoggingConfig(BaseModel):
    """日志配置。"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = "logs"


class ModelConfig(BaseModel):
    """模型架构超参数（训练期间不可修改）。"""

    vocab_size: int = Field(default=32000, ge=1000, le=200000)
    d_model: int = Field(default=768, ge=64, le=4096)
    n_layers: int = Field(default=14, ge=1, le=128)
    n_heads: int = Field(default=12, ge=1, le=64)
    d_ff: int = Field(default=3072, ge=256, le=32768)
    max_seq_len: int = Field(default=2048, ge=128, le=32768)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)


class OptimizerConfig(BaseModel):
    """优化器参数。"""

    name: Literal["adamw"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-8, gt=0.0)


class SchedulerConfig(BaseModel):
    """学习率调度器参数。"""

    name: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr: float = Field(default=3e-5, ge=0.0)


class TrainingConfig(BaseModel):
    """通用训练配置。"""

    batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    warmup_steps: int = Field(default=1000, ge=0)
    max_steps: int | None = None
    max_epochs: int | None = None
    eval_every: int = Field(default=500, ge=1)
    save_every: int = Field(default=2000, ge=1)
    max_checkpoints: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _check_step_epoch_mutual_exclusion(self) -> TrainingConfig:
        """max_steps 和 max_epochs 必须提供且仅提供一个。"""
        has_steps = self.max_steps is not None
        has_epochs = self.max_epochs is not None
        if has_steps == has_epochs:
            raise ValueError("必须提供 max_steps 或 max_epochs 中的一个（不可同时提供或同时省略）")
        return self


class DataConfig(BaseModel):
    """数据配置（SFT 使用）。"""

    max_samples: int = Field(default=15000, ge=1)
    val_split: float = Field(default=0.05, ge=0.0, le=1.0)


# ─── 顶层配置模型 ───────────────────────────────────────────────────────


class Settings(BaseSettings):
    """顶层配置基类 —— 对应 YAML 根节点。

    环境变量覆盖前缀: CCLLM_
    嵌套分隔符: __ (例如 CCLLM_TRAINING__BATCH_SIZE=16)
    """

    model_config = SettingsConfigDict(
        env_prefix="CCLLM_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    seed: int = Field(default=42, ge=0)
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class PretrainConfig(Settings):
    """预训练配置。"""

    training: TrainingConfig = Field(
        default_factory=lambda: TrainingConfig(
            batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,
            max_steps=100000,
        )
    )


class SFTConfig(Settings):
    """指令微调配置。"""

    training: TrainingConfig = Field(
        default_factory=lambda: TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            max_epochs=3,
        )
    )
    chat_template: str = "classical_chinese_v1"
    data: DataConfig = Field(default_factory=DataConfig)
