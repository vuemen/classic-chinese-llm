"""工具模块测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from classic_chinese_llm.utils.checkpoint import (
    CheckpointState,
    find_best_checkpoint,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from classic_chinese_llm.utils.device import DeviceInfo, detect_device, get_dtype, supports_bf16


class TestDeviceDetection:
    """设备检测测试。"""

    def test_detect_device_returns_device_info(self) -> None:
        """detect_device 始终返回 DeviceInfo。"""
        info = detect_device()
        assert isinstance(info, DeviceInfo)
        assert isinstance(info.device, torch.device)
        assert len(info.device_name) > 0

    def test_cpu_device_info(self) -> None:
        """CPU 设备的信息正确。"""
        info = detect_device(prefer="cpu")
        assert info.device == torch.device("cpu")
        assert info.cuda_available is False
        assert info.bf16_supported is False

    def test_supports_bf16_no_error(self) -> None:
        """supports_bf16 调用不报错。"""
        result = supports_bf16()
        assert isinstance(result, bool)

    def test_get_dtype_fallback_to_fp32(self) -> None:
        """无 GPU 时 get_dtype 回退到 fp32。"""
        info = detect_device(prefer="cpu")
        dtype = get_dtype(info, preference="bf16")
        assert dtype == torch.float32


class TestCheckpoint:
    """Checkpoint 管理测试。"""

    @staticmethod
    def _make_dummy_state(step: int = 100) -> CheckpointState:
        """创建测试用的 CheckpointState。"""
        return CheckpointState(
            model_state_dict={"weight": torch.randn(10, 10)},
            optimizer_state_dict={"param_groups": [{"lr": 0.001}]},
            global_step=step,
            epoch=1,
            best_loss=2.5,
            rng_state={"torch": torch.get_rng_state()},
            metadata={"train_loss": 2.5},
        )

    def test_save_and_load_roundtrip(self, temp_dir: Path) -> None:
        """保存后加载，数据一致。"""
        ckpt_dir = temp_dir / "checkpoints"
        original = self._make_dummy_state(step=500)

        saved_path = save_checkpoint(original, ckpt_dir, tag="step_500")
        assert saved_path.exists()

        loaded = load_checkpoint(saved_path, map_location="cpu")
        assert loaded.global_step == 500
        assert loaded.epoch == 1
        assert loaded.best_loss == 2.5
        assert torch.equal(loaded.model_state_dict["weight"], original.model_state_dict["weight"])
        assert loaded.optimizer_state_dict is not None
        assert loaded.metadata["train_loss"] == 2.5

    def test_save_creates_metadata_json(self, temp_dir: Path) -> None:
        """保存时同时生成 .json 元信息文件。"""
        ckpt_dir = temp_dir / "checkpoints"
        state = self._make_dummy_state()
        save_checkpoint(state, ckpt_dir, tag="test")

        meta_path = ckpt_dir / "checkpoint_test.json"
        assert meta_path.exists()
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["train_loss"] == 2.5

    def test_find_latest_in_empty_dir(self, temp_dir: Path) -> None:
        """空目录返回 None。"""
        result = find_latest_checkpoint(temp_dir / "empty")
        assert result is None

    def test_find_latest_with_files(self, temp_dir: Path) -> None:
        """存在文件时返回最新。"""
        ckpt_dir = temp_dir / "checkpoints"
        state = self._make_dummy_state()

        # 保存两个 checkpoint
        save_checkpoint(state, ckpt_dir, tag="step_100")
        state.global_step = 200
        save_checkpoint(state, ckpt_dir, tag="step_200")

        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None
        loaded = load_checkpoint(latest, map_location="cpu")
        assert loaded.global_step == 200

    def test_find_best_checkpoint(self, temp_dir: Path) -> None:
        """查找 best checkpoint。"""
        ckpt_dir = temp_dir / "checkpoints"
        state = self._make_dummy_state()
        save_checkpoint(state, ckpt_dir, tag="best")

        best = find_best_checkpoint(ckpt_dir)
        assert best is not None
        assert best.name == "checkpoint_best.pt"

    def test_load_nonexistent_raises(self, temp_dir: Path) -> None:
        """加载不存在的文件报错。"""
        with pytest.raises(FileNotFoundError):
            load_checkpoint(temp_dir / "nonexistent.pt", map_location="cpu")

    def test_rotate_old_checkpoints(self, temp_dir: Path) -> None:
        """旧 step checkpoint 被自动清理。"""
        ckpt_dir = temp_dir / "checkpoints"
        state = self._make_dummy_state()

        # 保存 7 个 step checkpoint，max_checkpoints=3
        for i in range(7):
            state.global_step = i * 100
            save_checkpoint(state, ckpt_dir, tag=f"step_{i*100}", max_checkpoints=3)

        # 只保留最近 3 个
        step_files = list(ckpt_dir.glob("checkpoint_step_*.pt"))
        assert len(step_files) == 3

    def test_best_checkpoint_not_rotated(self, temp_dir: Path) -> None:
        """best checkpoint 不被轮换清理。"""
        ckpt_dir = temp_dir / "checkpoints"
        state = self._make_dummy_state()

        save_checkpoint(state, ckpt_dir, tag="best")
        # 保存 10 个 step checkpoint，只保留 3 个
        for i in range(10):
            state.global_step = i * 100
            save_checkpoint(state, ckpt_dir, tag=f"step_{i*100}", max_checkpoints=3)

        assert (ckpt_dir / "checkpoint_best.pt").exists()
