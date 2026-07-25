"""pytest 全局 fixtures。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Path:
    """创建临时目录，测试结束后自动清理。"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def project_root(temp_dir: Path) -> Path:
    """模拟项目根目录（含基本目录结构）。"""
    (temp_dir / "configs").mkdir(exist_ok=True)
    (temp_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (temp_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (temp_dir / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (temp_dir / "models" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (temp_dir / "logs").mkdir(exist_ok=True)
    return temp_dir
