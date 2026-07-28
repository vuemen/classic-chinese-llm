"""PathConfig 单例路径管理器测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from classic_chinese_llm.config.paths import PathConfig


class TestPathConfigSingleton:
    """PathConfig 单例生命周期测试。"""

    def test_initialize_and_get(self, temp_dir: Path) -> None:
        """初始化后 get 返回正确的 PathConfig 实例。"""
        PathConfig.reset()
        PathConfig.initialize(temp_dir)
        paths = PathConfig.get()
        assert isinstance(paths, PathConfig)
        assert paths.root == temp_dir.resolve()

    def test_get_before_initialize_raises(self) -> None:
        """未初始化就调用 get 应抛出 RuntimeError。"""
        PathConfig.reset()
        with pytest.raises(RuntimeError, match="尚未初始化"):
            PathConfig.get()

    def test_reset_clears_singleton(self, temp_dir: Path) -> None:
        """reset 后 get 应再次抛出 RuntimeError。"""
        PathConfig.reset()
        PathConfig.initialize(temp_dir)
        assert PathConfig.get() is not None
        PathConfig.reset()
        with pytest.raises(RuntimeError):
            PathConfig.get()

    def test_reinitialize_with_different_root(self, temp_dir: Path) -> None:
        """重新 initialize 可切换至不同的项目根目录。"""
        PathConfig.reset()
        PathConfig.initialize(temp_dir)
        paths = PathConfig.get()
        assert paths.root == temp_dir.resolve()

        # 切换到另一个路径
        another = temp_dir / "another_project"
        another.mkdir()
        PathConfig.initialize(another)
        paths2 = PathConfig.get()
        assert paths2.root == another.resolve()

    def test_initialize_accepts_string_path(self, temp_dir: Path) -> None:
        """initialize 接受字符串路径参数。"""
        PathConfig.reset()
        PathConfig.initialize(str(temp_dir))
        paths = PathConfig.get()
        assert paths.root == temp_dir.resolve()


class TestPathConfigProperties:
    """PathConfig 路径属性测试。"""

    @pytest.fixture(autouse=True)
    def _setup(self, temp_dir: Path) -> None:
        """每个测试前重新初始化 PathConfig。"""
        PathConfig.reset()
        PathConfig.initialize(temp_dir)
        self._root = temp_dir.resolve()

    def test_root_property(self) -> None:
        """root 返回项目根目录。"""
        paths = PathConfig.get()
        assert paths.root == self._root

    def test_src_dir(self) -> None:
        """src_dir 返回 src/classic_chinese_llm 子目录。"""
        paths = PathConfig.get()
        assert paths.src_dir == self._root / "src" / "classic_chinese_llm"

    def test_data_dir(self) -> None:
        """data_dir 返回 data 子目录。"""
        paths = PathConfig.get()
        assert paths.data_dir == self._root / "data"

    def test_raw_data_dir(self) -> None:
        """raw_data_dir 返回 data/raw 子目录。"""
        paths = PathConfig.get()
        assert paths.raw_data_dir == self._root / "data" / "raw"

    def test_processed_data_dir(self) -> None:
        """processed_data_dir 返回 data/processed 子目录。"""
        paths = PathConfig.get()
        assert paths.processed_data_dir == self._root / "data" / "processed"

    def test_models_dir(self) -> None:
        """models_dir 返回 models 子目录。"""
        paths = PathConfig.get()
        assert paths.models_dir == self._root / "models"

    def test_checkpoint_dir(self) -> None:
        """checkpoint_dir 返回 models/checkpoints 子目录。"""
        paths = PathConfig.get()
        assert paths.checkpoint_dir == self._root / "models" / "checkpoints"

    def test_configs_dir(self) -> None:
        """configs_dir 返回 configs 子目录。"""
        paths = PathConfig.get()
        assert paths.configs_dir == self._root / "configs"

    def test_tokenizer_dir(self) -> None:
        """tokenizer_dir 返回 models/tokenizer 子目录。"""
        paths = PathConfig.get()
        assert paths.tokenizer_dir == self._root / "models" / "tokenizer"

    def test_logs_dir(self) -> None:
        """logs_dir 返回 logs 子目录。"""
        paths = PathConfig.get()
        assert paths.logs_dir == self._root / "logs"


class TestPathConfigAbsolutePaths:
    """验证所有路径属性均为绝对路径。"""

    def test_all_paths_are_absolute(self, temp_dir: Path) -> None:
        """所有路径属性返回的都是 resolved 之后的绝对路径。"""
        PathConfig.reset()
        PathConfig.initialize(temp_dir)
        paths = PathConfig.get()

        all_props = [
            paths.root,
            paths.src_dir,
            paths.data_dir,
            paths.raw_data_dir,
            paths.processed_data_dir,
            paths.models_dir,
            paths.checkpoint_dir,
            paths.configs_dir,
            paths.tokenizer_dir,
            paths.logs_dir,
        ]

        for p in all_props:
            assert p.is_absolute(), f"{p} 应为绝对路径"

    def test_paths_are_resolved(self, temp_dir: Path) -> None:
        """验证路径经过 resolve 处理（消除 .. 等符号）。"""
        PathConfig.reset()
        # 传入一个带 .. 的路径，验证被 resolve
        tricky = temp_dir / "foo" / ".."
        PathConfig.initialize(tricky)
        paths = PathConfig.get()
        assert paths.root == temp_dir.resolve()
        assert ".." not in str(paths.root)

    def test_all_paths_under_project_root(self, temp_dir: Path) -> None:
        """所有路径均以 project_root 为前缀。"""
        PathConfig.reset()
        PathConfig.initialize(temp_dir)
        paths = PathConfig.get()
        r = temp_dir.resolve()

        assert paths.src_dir.is_relative_to(r)
        assert paths.data_dir.is_relative_to(r)
        assert paths.raw_data_dir.is_relative_to(r)
        assert paths.processed_data_dir.is_relative_to(r)
        assert paths.models_dir.is_relative_to(r)
        assert paths.checkpoint_dir.is_relative_to(r)
        assert paths.configs_dir.is_relative_to(r)
        assert paths.tokenizer_dir.is_relative_to(r)
        assert paths.logs_dir.is_relative_to(r)
