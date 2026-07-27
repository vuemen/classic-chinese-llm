"""pytest 全局 fixtures。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# 将 src/ 加入 Python 搜索路径，使测试能 import classic_chinese_llm 而无需 pip install
_src_path = Path(__file__).resolve().parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


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
