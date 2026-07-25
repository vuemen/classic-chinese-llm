"""设备检测 —— GPU/CPU 可用性、显存、dtype 兼容性检查。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    """设备信息汇总。"""

    device: torch.device
    device_name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: tuple[int, int] | None
    bf16_supported: bool
    fp16_supported: bool
    cuda_available: bool


def detect_device(prefer: str = "cuda") -> DeviceInfo:
    """检测并返回最优可用设备。

    检测顺序: CUDA GPU → MPS (macOS) → CPU

    Args:
        prefer: 偏好设备类型 "cuda" | "mps" | "cpu"

    Returns:
        DeviceInfo 对象，包含完整的硬件能力信息
    """
    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / (1024**3)
        vram_free = torch.cuda.mem_get_info()[0] / (1024**3)

        return DeviceInfo(
            device=device,
            device_name=torch.cuda.get_device_name(0),
            vram_total_gb=round(vram_total, 1),
            vram_free_gb=round(vram_free, 1),
            compute_capability=(props.major, props.minor),
            bf16_supported=torch.cuda.is_bf16_supported(),
            fp16_supported=True,
            cuda_available=True,
        )

    if prefer in ("mps", "cuda") and torch.backends.mps.is_available():
        return DeviceInfo(
            device=torch.device("mps"),
            device_name="Apple MPS",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            compute_capability=None,
            bf16_supported=False,
            fp16_supported=True,
            cuda_available=False,
        )

    # Fallback: CPU
    return DeviceInfo(
        device=torch.device("cpu"),
        device_name="CPU",
        vram_total_gb=0.0,
        vram_free_gb=0.0,
        compute_capability=None,
        bf16_supported=False,
        fp16_supported=False,
        cuda_available=False,
    )


def supports_bf16() -> bool:
    """检测当前 CUDA 设备是否支持 BF16 混合精度训练。

    BF16 需要 Ampere 架构及以上（计算能力 ≥ 8.0）。
    """
    if not torch.cuda.is_available():
        return False
    return bool(torch.cuda.is_bf16_supported())


def get_dtype(device_info: DeviceInfo, preference: str = "bf16") -> torch.dtype:
    """根据设备能力和偏好选择最佳 dtype。

    优先级: bf16 > fp16 > fp32
    """
    if preference == "bf16" and device_info.bf16_supported:
        return torch.bfloat16
    if preference in ("bf16", "fp16") and device_info.fp16_supported:
        return torch.float16
    return torch.float32


def log_device_info(device_info: DeviceInfo) -> None:
    """打印格式化的设备信息报告（使用 rich markup）。"""
    from classic_chinese_llm.utils.logging_config import get_logger

    logger = get_logger(__name__)

    logger.info("[bold]设备检测报告[/bold]")
    logger.info("  设备: %s", device_info.device_name)
    if device_info.cuda_available:
        logger.info(
            "  显存: %.1f GB 总计 / %.1f GB 可用",
            device_info.vram_total_gb,
            device_info.vram_free_gb,
        )
        logger.info("  BF16: %s", "✅" if device_info.bf16_supported else "❌")
        logger.info("  FP16: %s", "✅" if device_info.fp16_supported else "❌")
