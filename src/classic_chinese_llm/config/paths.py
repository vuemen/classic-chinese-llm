"""路径常量 —— 基于项目根目录的路径解析单例。

使用方式:
    from classic_chinese_llm.config.paths import PathConfig

    PathConfig.initialize(project_root="/path/to/project")
    paths = PathConfig.get()
    print(paths.checkpoint_dir)  # /path/to/project/models/checkpoints
"""

from __future__ import annotations

from pathlib import Path


class PathConfig:
    """单例路径管理器。

    所有路径属性均基于 project_root 解析，确保路径一致性。
    测试中可以重新 initialize 来切换根目录。
    """

    _instance: PathConfig | None = None

    def __init__(self, project_root: str | Path) -> None:
        self._root = Path(project_root).resolve()

    # ─── 单例管理 ─────────────────────────────────────────────────

    @classmethod
    def initialize(cls, project_root: str | Path) -> PathConfig:
        """首次初始化路径管理器（应用启动时调用一次）。"""
        cls._instance = cls(project_root)
        return cls._instance

    @classmethod
    def get(cls) -> PathConfig:
        """获取已初始化的单例。"""
        if cls._instance is None:
            raise RuntimeError("PathConfig 尚未初始化，请先调用 PathConfig.initialize()")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例（用于测试）。"""
        cls._instance = None

    # ─── 路径属性 ─────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """项目根目录。"""
        return self._root

    @property
    def src_dir(self) -> Path:
        """源码目录。"""
        return self._root / "src" / "classic_chinese_llm"

    @property
    def data_dir(self) -> Path:
        """数据根目录。"""
        return self._root / "data"

    @property
    def raw_data_dir(self) -> Path:
        """原始数据目录。"""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """清洗后数据目录。"""
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        """模型根目录。"""
        return self._root / "models"

    @property
    def checkpoint_dir(self) -> Path:
        """训练 checkpoint 目录。"""
        return self.models_dir / "checkpoints"

    @property
    def configs_dir(self) -> Path:
        """配置文件目录。"""
        return self._root / "configs"

    @property
    def tokenizer_dir(self) -> Path:
        """Tokenizer 模型目录。"""
        return self.models_dir / "tokenizer"

    @property
    def logs_dir(self) -> Path:
        """日志目录。"""
        return self._root / "logs"
