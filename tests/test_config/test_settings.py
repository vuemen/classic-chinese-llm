"""配置模块测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from classic_chinese_llm.config import (
    PretrainConfig,
    SFTConfig,
    load_config,
)
from classic_chinese_llm.config.paths import PathConfig
from classic_chinese_llm.config.settings import (
    ModelConfig,
    TrainingConfig,
)


class TestTrainingConfig:
    """TrainingConfig 校验测试。"""

    def test_max_steps_and_epochs_mutual_exclusion_steps(self) -> None:
        """仅提供 max_steps 时校验通过。"""
        cfg = TrainingConfig(max_steps=1000)
        assert cfg.max_steps == 1000
        assert cfg.max_epochs is None

    def test_max_steps_and_epochs_mutual_exclusion_epochs(self) -> None:
        """仅提供 max_epochs 时校验通过。"""
        cfg = TrainingConfig(max_epochs=10)
        assert cfg.max_epochs == 10
        assert cfg.max_steps is None

    def test_both_provided_raises(self) -> None:
        """同时提供两者应报错。"""
        with pytest.raises(ValueError, match="必须提供 max_steps 或 max_epochs"):
            TrainingConfig(max_steps=1000, max_epochs=10)

    def test_neither_provided_raises(self) -> None:
        """都不提供应报错。"""
        with pytest.raises(ValueError, match="必须提供 max_steps 或 max_epochs"):
            TrainingConfig()


class TestModelConfig:
    """ModelConfig 校验测试。"""

    def test_default_values(self) -> None:
        """验证默认值与架构文档一致。"""
        cfg = ModelConfig()
        assert cfg.vocab_size == 32000
        assert cfg.d_model == 768
        assert cfg.n_layers == 14
        assert cfg.n_heads == 12
        assert cfg.d_ff == 3072
        assert cfg.max_seq_len == 2048

    def test_d_model_out_of_range_raises(self) -> None:
        """d_model 超出范围报错。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelConfig(d_model=100_000)


class TestPretrainConfig:
    """PretrainConfig 加载测试。"""

    def test_load_from_yaml(self, project_root: Path) -> None:
        """从 YAML 文件加载预训练配置。"""
        PathConfig.reset()
        PathConfig.initialize(project_root)

        # 写入 default.yaml
        (project_root / "configs").mkdir(exist_ok=True)
        default = {
            "seed": 42,
            "dtype": "bf16",
            "logging": {"level": "INFO", "log_dir": "logs"},
            "model": {
                "vocab_size": 32000,
                "d_model": 768,
                "n_layers": 14,
                "n_heads": 12,
                "d_ff": 3072,
                "max_seq_len": 2048,
            },
            "training": {"max_steps": 1000},
            "optimizer": {"name": "adamw"},
            "scheduler": {"name": "cosine"},
        }
        with open(project_root / "configs" / "default.yaml", "w") as f:
            yaml.dump(default, f)

        config = load_config(project_root / "configs" / "default.yaml", PretrainConfig)
        assert config.seed == 42
        assert config.dtype == "bf16"
        assert config.model.d_model == 768
        assert config.training.max_steps == 1000

    def test_yaml_inheritance(self, project_root: Path) -> None:
        """测试 YAML extends 继承 + 覆盖。"""
        PathConfig.reset()
        PathConfig.initialize(project_root)

        base = {
            "seed": 42,
            "dtype": "bf16",
            "model": {"d_model": 768},
            "training": {"max_steps": 1000},
            "logging": {"level": "INFO"},
        }
        with open(project_root / "configs" / "base.yaml", "w") as f:
            yaml.dump(base, f)

        child = {
            "extends": "base.yaml",
            "seed": 99,
            "model": {"d_model": 1024},  # 覆盖 base
        }
        child_path = project_root / "configs" / "child.yaml"
        with open(child_path, "w") as f:
            yaml.dump(child, f)

        config = load_config(child_path, PretrainConfig)
        assert config.seed == 99  # 被子配置覆盖
        assert config.model.d_model == 1024  # 被子配置覆盖
        assert config.training.max_steps == 1000  # 从父配置继承
        assert config.dtype == "bf16"  # 从父配置继承


class TestSFTConfig:
    """SFTConfig 加载测试。"""

    def test_default_values(self) -> None:
        """验证 SFT 默认值——未显式传参时使用 Field default_factory 的值。"""
        config = SFTConfig()
        assert config.training.max_epochs == 3
        assert config.training.batch_size == 4
        assert config.training.gradient_accumulation_steps == 8
        assert config.training.learning_rate == 1e-4
        assert config.chat_template == "classical_chinese_v1"
        assert config.data.max_samples == 15000
        assert config.data.val_split == 0.05

    def test_training_override(self) -> None:
        """显式传入 training 字段时覆盖默认值（未传字段继承 TrainingConfig 默认值）。"""
        config = SFTConfig(
            training={"max_epochs": 10, "batch_size": 2},  # type: ignore[arg-type]
        )
        assert config.training.max_epochs == 10
        assert config.training.batch_size == 2
        assert config.chat_template == "classical_chinese_v1"


class TestPathConfig:
    """PathConfig 路径管理测试。"""

    def test_initialize_and_get(self, temp_dir: Path) -> None:
        """初始化后 get 返回正确实例。"""
        PathConfig.reset()
        PathConfig.initialize(temp_dir)
        paths = PathConfig.get()
        assert paths.root == temp_dir.resolve()

    def test_get_before_initialize_raises(self) -> None:
        """未初始化就 get 应报错。"""
        PathConfig.reset()
        with pytest.raises(RuntimeError, match="尚未初始化"):
            PathConfig.get()

    def test_path_properties(self, temp_dir: Path) -> None:
        """验证各路径属性正确拼接。"""
        PathConfig.reset()
        PathConfig.initialize(temp_dir)
        paths = PathConfig.get()

        assert paths.data_dir == temp_dir.resolve() / "data"
        assert paths.checkpoint_dir == temp_dir.resolve() / "models" / "checkpoints"
        assert paths.configs_dir == temp_dir.resolve() / "configs"

    def test_reset(self, temp_dir: Path) -> None:
        """reset 后 get 应报错。"""
        PathConfig.reset()
        PathConfig.initialize(temp_dir)
        assert PathConfig.get() is not None
        PathConfig.reset()
        with pytest.raises(RuntimeError):
            PathConfig.get()
