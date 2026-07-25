# 工具模块设计文档

**所属阶段:** Phase 1 — 基础设施
**涉及模块:** `src/classic_chinese_llm/utils/`
**日期:** 2026-07-25

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | 结构化日志 | 同时输出到控制台（rich 彩色格式）和文件（纯文本），支持模块级 logger |
| F2 | 设备检测 | 自动检测 GPU/CPU、显存容量、BF16/FP16 兼容性 |
| F3 | Checkpoint 保存 | 保存模型权重 + 优化器状态 + 训练元信息（step、epoch、loss、RNG 状态） |
| F4 | Checkpoint 恢复 | 从 checkpoint 完整恢复训练状态，支持中断续训 |
| F5 | Checkpoint 管理 | 自动清理旧 checkpoint，保留最近 N 个 + 最佳模型 |

### 1.2 非功能需求

- **零额外依赖**: 仅使用 `torch`、`logging`、`rich`（已作为项目依赖）
- **原子写入**: Checkpoint 必须原子保存（要么全写完，要么不写），避免中断导致文件损坏
- **可恢复性**: 训练中断后，从 checkpoint 恢复的 loss 曲线应与中断前连续

---

## 2. 方案选型与对比

### 2.1 日志方案

| 方案 | 彩色输出 | 文件日志 | 结构化 | 额外依赖 | 结论 |
|------|----------|----------|--------|----------|------|
| **stdlib logging + rich** | ✅ RichHandler | ✅ FileHandler | ✅ 手动格式 | 0（rich 已是依赖） | ✅ 选用 |
| loguru | ✅ 原生 | ✅ 原生 | ✅ 原生 | loguru | ❌ |
| structlog | ❌ 需配置 | ❌ 需配置 | ✅ 原生 | structlog | ❌ |
| print() | ❌ | ❌ | ❌ | 0 | ❌ 禁止 |

**详细对比**:

**loguru**: API 简洁（`logger.info("hello {name}", name="world")`），但引入额外依赖。它本质是对 stdlib logging 的封装——对于本项目，直接使用 stdlib + rich Handler 可以达到同样效果，零增量成本。

**structlog**: 结构化日志的事实标准，但本项目不是分布式系统，不需要 JSON 格式日志的机器可读性。对单机训练场景来说过于重量级。

**stdlib logging + rich（最终选择）**: Python 标准库的 logging 模块虽然 API 稍显冗长，但配置一次即可全项目使用。rich 的 `RichHandler` 提供了开箱即用的彩色格式、traceback 美化、列对齐。

### 2.2 设备检测方案

| 方案 | GPU 信息 | 显存 | BF16 检测 | 额外依赖 | 结论 |
|------|----------|------|-----------|----------|------|
| **torch.cuda API** | ✅ | ✅ | ✅ | 0 | ✅ 选用 |
| pynvml | ✅ 更详细 | ✅ | ❌ 需 torch | pynvml | ❌ |
| GPUtil | ✅ | ✅ | ❌ | GPUtil | ❌ |

**最终选择**: **torch.cuda API**。项目已经依赖 PyTorch，`torch.cuda` 提供了所有需要的信息：

```python
torch.cuda.is_available()        # GPU 是否可用
torch.cuda.get_device_name(0)    # GPU 型号
torch.cuda.get_device_properties(0)  # 显存、计算能力
torch.cuda.is_bf16_supported()   # BF16 支持
```

### 2.3 Checkpoint 格式

| 方案 | 安全性 | 跨框架 | 加载速度 | 实现复杂度 | 结论 |
|------|--------|--------|----------|------------|------|
| **torch.save/load** | 中（pickle 兼容性风险） | ❌ 仅 PyTorch | 快 | 低 | ✅ 选用 |
| safetensors | 高（无代码执行风险） | ✅ 多框架 | 中 | 中 | ❌ 不必要 |
| HF 格式 (bin + config.json) | 中 | ✅ HF 生态 | 中 | 高 | ❌ 违背"零 HF 模型代码"原则 |

**详细对比**:

**safetensors**: 优势是安全（不执行 pickle 代码）和跨框架。但本项目是 PyTorch 独占项目，不跨框架分发模型；且 safetensors 对自定义模型支持有限（通常需要配合 HF 的 `model.config`）。

**torch.save/load（最终选择）**: 对于从零实现的自定义模型，`torch.save` 最简单直接。风险在于 Python/PyTorch 版本升级可能导致旧 pickle 无法加载——通过固定 `requires-python` 和 `torch>=2.4` 来管理。

**为什么不选 safetensors**: 本项目约束是"不依赖 HF 模型代码"。safetensors 需要手动将每个参数的 tensor 映射到 key——本质上等于实现一套序列化协议。在单人单机单框架的前提下，`torch.save` 是务实的选择。

---

## 3. 最终方案

### 3.1 模块结构

```
src/classic_chinese_llm/utils/
├── __init__.py              # 统一导出
├── logging_config.py        # 日志系统初始化
├── device.py                # 设备检测与报告
└── checkpoint.py            # Checkpoint 保存/恢复/管理
```

### 3.2 日志系统详细设计

```python
# logging_config.py

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    *,
    rich_width: int = 120,
) -> None:
    """
    初始化全局日志配置。

    配置 root logger，所有模块通过 logging.getLogger(__name__) 自动继承。

    输出:
    - 终端: RichHandler（彩色、traceback 美化、列对齐）
    - 文件: FileHandler（纯文本，时间戳 + 模块名 + 级别 + 消息）

    Args:
        level: 日志级别 DEBUG | INFO | WARNING | ERROR
        log_file: 日志文件路径，None 表示不输出到文件
        rich_width: Rich 控制台宽度
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))

    # ── 终端 Handler ──────────────────────────────────
    console = Console(width=rich_width)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,         # 不显示文件路径以节省空间
        rich_tracebacks=True,    # rich 美化异常堆栈
        markup=True,             # 支持 rich markup 语法
    )
    rich_handler.setLevel(logging.DEBUG)
    # 终端格式：仅消息，RichHandler 自行处理时间/级别
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rich_handler)

    # ── 文件 Handler ──────────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(file_handler)

    # ── 抑制第三方库噪音 ───────────────────────────────
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取模块级 logger。
    使用方式: logger = get_logger(__name__)
    """
    return logging.getLogger(name)
```

**使用示例**:

```python
# 任意模块
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

def train():
    logger.info("开始训练 | model=%s params=%d", "classical-llm", 157_000_000)
    logger.debug("batch shape: %s", (8, 2048))
    try:
        loss = forward(batch)
    except Exception:
        logger.exception("训练步骤失败")  # RichHandler 自动美化 traceback
```

### 3.3 设备检测详细设计

```python
# device.py

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceInfo:
    """设备信息汇总"""
    device: torch.device
    device_name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: tuple[int, int] | None
    bf16_supported: bool
    fp16_supported: bool
    cuda_available: bool


def detect_device(prefer: str = "cuda") -> DeviceInfo:
    """
    检测并返回最优可用设备。

    检测顺序: CUDA GPU → MPS (macOS) → CPU
    返回 DeviceInfo 包含完整的硬件能力报告。

    Args:
        prefer: 偏好设备 "cuda" | "mps" | "cpu"

    Returns:
        DeviceInfo 设备信息对象
    """
    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_mem / (1024 ** 3)
        vram_free = (torch.cuda.mem_get_info()[0]) / (1024 ** 3)

        return DeviceInfo(
            device=device,
            device_name=torch.cuda.get_device_name(0),
            vram_total_gb=round(vram_total, 1),
            vram_free_gb=round(vram_free, 1),
            compute_capability=(props.major, props.minor),
            bf16_supported=torch.cuda.is_bf16_supported(),
            fp16_supported=True,  # CUDA GPU 都支持 FP16
            cuda_available=True,
        )

    if prefer in ("mps", "cuda") and torch.backends.mps.is_available():
        return DeviceInfo(
            device=torch.device("mps"),
            device_name="Apple MPS",
            vram_total_gb=0.0,    # MPS 不提供显存信息
            vram_free_gb=0.0,
            compute_capability=None,
            bf16_supported=False, # MPS BF16 支持有限
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
    """检测当前设备是否支持 BF16 混合精度训练"""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_bf16_supported()


def get_dtype(device_info: DeviceInfo, preference: str = "bf16") -> torch.dtype:
    """
    根据设备能力选择最佳 dtype。

    优先级: bf16 > fp16 > fp32
    实测 BF16 在 RTX 30 系列及以上支持，覆盖 12GB VRAM 的典型配置。
    """
    if preference == "bf16" and device_info.bf16_supported:
        return torch.bfloat16
    if preference in ("bf16", "fp16") and device_info.fp16_supported:
        return torch.float16
    return torch.float32


def log_device_info(device_info: DeviceInfo) -> None:
    """
    打印格式化的设备信息报告（使用 rich markup）。

    示例输出:
    ┌─ GPU 设备信息 ───────────────────
    │ 设备名称: NVIDIA RTX 4070
    │ 显存总计: 11.8 GB
    │ 显存可用: 10.2 GB
    │ 计算能力: 8.9
    │ BF16 支持: ✅
    │ FP16 支持: ✅
    └──────────────────────────────────
    """
    from classic_chinese_llm.utils.logging_config import get_logger
    logger = get_logger(__name__)

    logger.info("[bold]设备检测报告[/bold]")
    logger.info("  设备: %s", device_info.device_name)
    if device_info.cuda_available:
        logger.info("  显存: %.1f GB 总计 / %.1f GB 可用",
                     device_info.vram_total_gb, device_info.vram_free_gb)
        logger.info("  BF16: %s", "✅" if device_info.bf16_supported else "❌")
        logger.info("  FP16: %s", "✅" if device_info.fp16_supported else "❌")
```

### 3.4 Checkpoint 管理详细设计

```python
# checkpoint.py

import json
import shutil
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# ─── 数据结构 ─────────────────────────────────────────

@dataclass
class CheckpointState:
    """Checkpoint 包含的完整训练状态"""
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Optional[Dict[str, Any]]
    global_step: int
    epoch: int
    best_loss: float
    rng_state: Optional[Dict[str, Any]]   # torch + numpy + python RNG 状态
    metadata: Dict[str, Any]              # 用户自定义元信息


# ─── 保存 ─────────────────────────────────────────────

def save_checkpoint(
    state: CheckpointState,
    checkpoint_dir: str | Path,
    *,
    tag: str = "step",
    max_checkpoints: int = 5,
) -> Path:
    """
    原子保存 checkpoint。

    流程:
    1. 写入临时文件 checkpoint_tmp.pt + metadata_tmp.json
    2. os.rename 原子重命名为最终文件（POSIX 原子性，Windows 尽力）
    3. 清理超过 max_checkpoints 的旧 checkpoint

    Args:
        state: CheckpointState 训练状态
        checkpoint_dir: checkpoint 目录
        tag: 文件标签，如 "step_1000" 或 "best"
        max_checkpoints: 保留最近 N 个 checkpoint

    Returns:
        保存的 .pt 文件路径
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = ckpt_dir / f"checkpoint_{tag}.pt"
    meta_file = ckpt_dir / f"checkpoint_{tag}.json"

    # 构建保存对象
    checkpoint = {
        "model_state_dict": state.model_state_dict,
        "optimizer_state_dict": state.optimizer_state_dict,
        "global_step": state.global_step,
        "epoch": state.epoch,
        "best_loss": state.best_loss,
        "rng_state": state.rng_state,
    }

    # 原子写入模型权重（通过临时文件 + rename）
    tmp_ckpt = ckpt_dir / f".checkpoint_{tag}.tmp"
    tmp_meta = ckpt_dir / f".checkpoint_{tag}_meta.tmp"

    try:
        torch.save(checkpoint, tmp_ckpt)
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(state.metadata, f, ensure_ascii=False, indent=2)

        # 原子重命名（同文件系统内 rename 在 POSIX 上是原子的）
        tmp_ckpt.replace(ckpt_file)
        tmp_meta.replace(meta_file)

    finally:
        # 清理可能的临时文件
        tmp_ckpt.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)

    logger.info("Checkpoint 已保存: %s (step=%d)", ckpt_file, state.global_step)

    # 清理旧 checkpoint
    _rotate_checkpoints(checkpoint_dir, keep=max_checkpoints)

    return ckpt_file


# ─── 加载 ─────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cuda",
) -> CheckpointState:
    """
    加载 checkpoint 并返回 CheckpointState。

    Args:
        checkpoint_path: .pt 文件路径
        map_location: torch.load 的设备映射

    Returns:
        CheckpointState（optimizer_state_dict 可能为 None）

    Raises:
        FileNotFoundError: checkpoint 文件不存在
        KeyError: checkpoint 文件格式不完整
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 文件不存在: {ckpt_path}")

    logger.info("正在加载 checkpoint: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    # 加载元信息（可选，不存在不报错）
    meta_path = ckpt_path.with_suffix(".json")
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    state = CheckpointState(
        model_state_dict=checkpoint["model_state_dict"],
        optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
        global_step=checkpoint["global_step"],
        epoch=checkpoint.get("epoch", 0),
        best_loss=checkpoint.get("best_loss", float("inf")),
        rng_state=checkpoint.get("rng_state"),
        metadata=metadata,
    )

    logger.info("Checkpoint 加载完成 | step=%d epoch=%d best_loss=%.4f",
                 state.global_step, state.epoch, state.best_loss)
    return state


# ─── 查找 ─────────────────────────────────────────────

def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """
    在目录中查找最新的 checkpoint。

    规则:
    1. 优先找 "checkpoint_latest.pt"
    2. 其次按文件修改时间排序，找最近的非 best checkpoint
    3. 最后找 checkpoint_best.pt

    Returns:
        最新 checkpoint 路径，无文件返回 None
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    # 优先：latest symlink
    latest = ckpt_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest

    # 其次：按修改时间找最新的 step checkpoint
    checkpoints = sorted(
        ckpt_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if checkpoints:
        return checkpoints[0]

    # 最后：best checkpoint
    best = ckpt_dir / "checkpoint_best.pt"
    if best.exists():
        return best

    return None


def find_best_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """查找 best checkpoint"""
    best = Path(checkpoint_dir) / "checkpoint_best.pt"
    return best if best.exists() else None


# ─── 内部函数 ─────────────────────────────────────────

def _rotate_checkpoints(checkpoint_dir: str | Path, keep: int) -> None:
    """
    清理旧 checkpoint，保留最近 N 个 step checkpoint。

    不删除:
    - checkpoint_best.pt（最佳模型永久保留）
    - checkpoint_latest.pt（最新模型）
    """
    ckpt_dir = Path(checkpoint_dir)
    step_checkpoints = sorted(
        ckpt_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    for old in step_checkpoints[:-max(0, keep)]:
        logger.debug("清理旧 checkpoint: %s", old)
        old.unlink(missing_ok=True)
        # 同时清理配套的 json 元信息
        old.with_suffix(".json").unlink(missing_ok=True)
```

---

## 4. 关键技术点

### 4.1 RichHandler 的 traceback 美化

`rich.traceback.install()` + `RichHandler(rich_tracebacks=True)` 组合提供比纯文本 traceback 更可读的异常输出：

- 本地变量自动展开（限制深度）
- 错误行高亮
- 链式异常（`raise ... from ...`）以嵌套方式展示

这在调试训练崩溃时极其有用。

### 4.2 Checkpoint 原子写入

训练中途崩溃（OOM、断电、CUDA error）时，正在写入的 checkpoint 可能只写了一半。使用"临时文件 + rename"模式确保原子性：

```
1. torch.save → .checkpoint_step_1000.tmp
2. os.replace → checkpoint_step_1000.pt  (原子操作)
```

`os.replace` 在同文件系统上是原子操作（POSIX 保证，Windows NTFS 也支持）。要么旧文件存在（完整），要么新文件存在（完整），不存在半个文件状态。

### 4.3 BF16 检测逻辑

```
RTX 30 系列 (Ampere, SM 8.0/8.6)  → 支持 BF16
RTX 20 系列 (Turing, SM 7.5)     → 不支持 BF16
GTX 16 系列 (Turing, SM 7.5 CUT) → 不支持 BF16
V100 (Volta, SM 7.0)             → 不支持 BF16
```

`torch.cuda.is_bf16_supported()` 内部检查计算能力 ≥ 8.0。BF16 对训练稳定性的价值很大（动态范围与 FP32 相同），因此设备检测报告中 BF16 支持是重点信息。

### 4.4 Checkpoint 中的 RNG 状态

完整恢复训练需要保存三类随机状态：

```python
rng_state = {
    "torch": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all(),  # 所有 GPU
    "numpy": np.random.get_state(),
    "python": random.getstate(),
}
```

不保存 RNG 状态的后果：续训时的 dropout mask、数据 shuffle 顺序与中断前不同，导致 loss 曲线不连续（虽然不影响最终收敛）。

### 4.5 `weights_only=True` 安全性

PyTorch 2.4+ 支持 `torch.load(..., weights_only=True)`。开启后只反序列化 tensor，拒绝任意 Python 对象。配合 `torch.serialization.add_safe_globals` 可以安全加载自定义类型。

```python
# 仅在加载他人提供的 checkpoint 时有安全收益
# 本项目场景为加载自己保存的 checkpoint，但仍保持好习惯
checkpoint = torch.load(ckpt_path, weights_only=True)
```

---

## 5. 与其他模块的关系

```
Utils ─── 被依赖 ───> Training (Trainer 使用 checkpoint、logging、device)
Utils ─── 被依赖 ───> Inference (Engine 使用 checkpoint、device)
Utils ─── 被依赖 ───> Data (Collector 使用 logging)
Utils ─── 依赖 ───> Config (logging_config 接受 LoggingConfig 参数)
```

Utils 模块是横切关注点（cross-cutting concern），被几乎所有上层模块依赖。但 Utils 内部三个组件之间**相互独立**——logging、device、checkpoint 之间没有直接依赖。

---

## 6. 验证清单

- [ ] `setup_logging(level="INFO", log_file="train.log")` 正常输出到终端和文件
- [ ] `detect_device()` 在无 GPU 环境正确 fallback 到 CPU
- [ ] `detect_device()` 在 12GB GPU 环境报告正确的 VRAM 信息
- [ ] `save_checkpoint()` → `load_checkpoint()` 往返一致（权重、optimizer 状态、step）
- [ ] Checkpoint 原子写入：写入过程中断不损坏已有 checkpoint
- [ ] `_rotate_checkpoints()` 正确清理旧文件，保留 best 和最新
- [ ] `find_latest_checkpoint()` 在空目录返回 None
