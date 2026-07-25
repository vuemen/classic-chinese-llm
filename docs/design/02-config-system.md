# 配置系统设计文档

**所属阶段:** Phase 1 — 基础设施
**涉及模块:** `src/classic_chinese_llm/config/`
**日期:** 2026-07-25

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | 类型安全配置 | IDE 自动补全、类型检查，拼写错误在开发时即可发现 |
| F2 | YAML 文件加载 | 从 `configs/` 目录加载 YAML，支持层级继承（default → pretrain → sft） |
| F3 | 环境变量覆盖 | 任何配置项均可通过环境变量覆盖（如 `CCLLM_SEED=123`） |
| F4 | 路径解析 | 项目内路径（data/、models/、checkpoints/）自动解析为绝对路径 |
| F5 | 配置校验 | 启动时验证配置合法性（值范围、互斥约束），避免跑到一半才报错 |

### 1.2 非功能需求

- **启动快速**: 配置加载 + 校验 < 100ms
- **错误信息友好**: 配置错误时明确指出哪个文件、哪个字段、期望什么值
- **可扩展**: 新增配置类新增字段不影响现有代码

---

## 2. 方案选型与对比

### 2.1 配置管理库

这是最核心的技术决策。

| 方案 | 类型安全 | 校验 | YAML | 环境变量 | 学习成本 | 结论 |
|------|----------|------|------|----------|----------|------|
| **Pydantic** | ✅ 原生 | ✅ 强大 | ✅ 需 yaml 解析 | ✅ pydantic-settings | 低 | ✅ 选用 |
| dataclasses | ✅ 原生 | ❌ 需手写 | ❌ 需手写 | ❌ 需手写 | 最低 | ❌ 功能不足 |
| OmegaConf | ❌ 运行时 | ✅ 结构化 | ✅ 原生 | ✅ 变量插值 | 中 | ❌ 无类型安全 |
| Hydra | ❌ 运行时 | ❌ 需额外 | ✅ 原生 | ✅ 原生 | 高 | ❌ 过度设计 |

**详细对比**:

**dataclasses + 手动解析**:
```python
# 类型安全，但校验、env 覆盖都要手写
@dataclass
class TrainConfig:
    lr: float
    batch_size: int

# 手动加载：20+ 行 boilerplate
def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # 手动类型转换、校验、环境变量覆盖...
```
❌ 缺点：校验逻辑散落、环境变量覆盖要手写、无嵌套配置校验。

**OmegaConf**:
```python
# 结构化但不类型安全
cfg = OmegaConf.load("config.yaml")
print(cfg.training.learning_rrate)  # 拼写错误！运行时才发现
```
❌ 缺点：无 IDE 补全、无编译期类型检查。

**Hydra**:
- 功能强大但引入了命令行解析、多任务运行等大量概念
- 主要用于研究实验管理（hyperparameter sweeping），本项目只需简单配置
- 引入 `@hydra.main` 装饰器改变了脚本入口，与 Claude Code 指引的 CLI 设计冲突

**Pydantic（最终选择）**:
```python
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=8, ge=1)

# IDE 自动补全 training_config.learning_rate
# 类型错误在开发时即被 mypy 捕获
```
✅ 优势：类型安全、内置校验（`gt`、`ge`、`validator`）、环境变量通过 `pydantic-settings` 无缝支持。

### 2.2 配置格式

| 格式 | 可读性 | 注释 | 嵌套 | PyTorch 生态 | 结论 |
|------|--------|------|------|-------------|------|
| **YAML** | ⭐⭐⭐ | ✅ | ✅ | 标准 | ✅ 选用 |
| TOML | ⭐⭐ | ✅ | ⭐ | 少用 | ❌ |
| JSON | ⭐ | ❌ | ✅ | 部分 | ❌ |

**最终选择**: **YAML**。Torch 生态（包括 HuggingFace 的 `training_args.yaml`）以 YAML 为标准，支持注释、多行字符串、锚点引用等实用特性。

### 2.3 环境变量覆盖方案

| 方案 | 实现方式 | 优势 | 劣势 |
|------|----------|------|------|
| **pydantic-settings** | `BaseSettings` + `env_prefix` | 与 Pydantic 深度集成，一行配置即可 | 额外依赖（但项目已引入 pydantic） |
| 手动 `os.environ.get` | 每个字段手动检查 | 零依赖 | boilerplate 多，易遗漏 |
| python-dotenv | `.env` 文件 + 手动注入 | .env 文件支持 | 仍需手动映射到字段 |

**最终选择**: **pydantic-settings**。已有 pydantic 依赖，`pydantic-settings` 增加几乎零成本。

---

## 3. 最终方案

### 3.1 模块结构

```
src/classic_chinese_llm/config/
├── __init__.py          # 导出 load_config、Settings、PathConfig
├── settings.py          # Pydantic 配置模型
└── paths.py             # 路径常量与解析
```

### 3.2 配置模型层级

```
BaseModel (Pydantic)
  └── Settings                # 通用配置基类
        ├── PretrainConfig    # 预训练配置
        ├── SFTConfig         # 指令微调配置
        └── EvalConfig        # 评测配置（后续阶段）
```

### 3.3 核心接口设计

```python
# settings.py

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional

# ─── 叶子配置模型 ───────────────────────────────────────

class LoggingConfig(BaseModel):
    """日志配置"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = "logs"

class ModelConfig(BaseModel):
    """模型架构参数（不可训练修改）"""
    vocab_size: int = Field(default=32000, ge=1000, le=200000)
    d_model: int = Field(default=768, ge=64, le=4096)
    n_layers: int = Field(default=14, ge=1, le=128)
    n_heads: int = Field(default=12, ge=1, le=64)
    d_ff: int = Field(default=3072, ge=256, le=32768)
    max_seq_len: int = Field(default=2048, ge=128, le=32768)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)

class OptimizerConfig(BaseModel):
    """优化器参数"""
    name: Literal["adamw"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-8, gt=0)

class SchedulerConfig(BaseModel):
    """学习率调度器参数"""
    name: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr: float = Field(default=3e-5, ge=0)

class TrainingConfig(BaseModel):
    """通用训练配置"""
    batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    warmup_steps: int = Field(default=1000, ge=0)
    max_steps: Optional[int] = None    # 与 max_epochs 互斥
    max_epochs: Optional[int] = None
    eval_every: int = Field(default=500, ge=1)
    save_every: int = Field(default=2000, ge=1)
    max_checkpoints: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def check_step_epoch_mutual_exclusion(self):
        """max_steps 和 max_epochs 必须提供一个且仅一个"""
        if (self.max_steps is None) == (self.max_epochs is None):
            raise ValueError("必须提供 max_steps 或 max_epochs 中的一个")
        return self

class DataConfig(BaseModel):
    """数据配置（SFT 使用）"""
    max_samples: int = Field(default=15000, ge=1)
    val_split: float = Field(default=0.05, ge=0.0, le=1.0)

# ─── 顶层配置模型 ───────────────────────────────────────

class Settings(BaseSettings):
    """顶层配置，对应 YAML 根节点"""

    model_config = SettingsConfigDict(
        env_prefix="CCLLM_",      # 环境变量前缀
        env_nested_delimiter="__", # 嵌套配置分隔符
        extra="forbid",            # 禁止未知字段
    )

    seed: int = Field(default=42, ge=0)
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig                  # 子类必须提供
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

class PretrainConfig(Settings):
    """预训练配置"""
    training: TrainingConfig = Field(
        default_factory=lambda: TrainingConfig(
            batch_size=8, gradient_accumulation_steps=4,
            learning_rate=3e-4, max_steps=100000,
        )
    )

class SFTConfig(Settings):
    """指令微调配置"""
    training: TrainingConfig = Field(
        default_factory=lambda: TrainingConfig(
            batch_size=4, gradient_accumulation_steps=8,
            learning_rate=1e-4, max_epochs=3,
        )
    )
    chat_template: str = "classical_chinese_v1"
    data: DataConfig = Field(default_factory=DataConfig)
```

### 3.4 路径管理

```python
# paths.py

import os
from pathlib import Path
from functools import lru_cache

class PathConfig:
    """路径常量——单例模式，基于项目根目录解析"""

    _instance: "PathConfig | None" = None

    def __init__(self, project_root: str | Path):
        self._root = Path(project_root).resolve()

    @classmethod
    def initialize(cls, project_root: str | Path) -> "PathConfig":
        """首次初始化（在应用启动时调用一次）"""
        cls._instance = cls(project_root)
        return cls._instance

    @classmethod
    def get(cls) -> "PathConfig":
        """获取单例（初始化后使用）"""
        if cls._instance is None:
            raise RuntimeError("PathConfig 尚未初始化，请先调用 initialize()")
        return cls._instance

    # ─── 路径属性 ─────────────────────────────────────

    @property
    def root(self) -> Path:
        return self._root

    @property
    def src_dir(self) -> Path:
        return self._root / "src" / "classic_chinese_llm"

    @property
    def data_dir(self) -> Path:
        return self._root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self._root / "models"

    @property
    def checkpoint_dir(self) -> Path:
        return self.models_dir / "checkpoints"

    @property
    def configs_dir(self) -> Path:
        return self._root / "configs"

    @property
    def tokenizer_dir(self) -> Path:
        return self.models_dir / "tokenizer"
```

### 3.5 配置加载器

```python
# __init__.py — 核心加载逻辑

import os
import yaml
from pathlib import Path
from typing import TypeVar

T = TypeVar("T", bound="Settings")

def load_config(
    config_path: str | Path,
    config_cls: type[T] = Settings,  # type: ignore[assignment]
) -> T:
    """
    统一配置加载流程:

    1. 解析 YAML（处理 extends 继承）
    2. 合并环境变量覆盖
    3. Pydantic 校验并返回强类型对象

    Args:
        config_path: YAML 配置文件路径（相对或绝对）
        config_cls: 期望的配置类型（默认 Settings）

    Returns:
        校验后的配置对象
    """
    raw = _load_yaml_with_inheritance(config_path)
    # pydantic-settings 自动将 CCLLM_* 环境变量注入对应字段
    return config_cls(**raw)

def _load_yaml_with_inheritance(config_path: str | Path) -> dict:
    """
    递归加载 YAML，支持 extends 继承。

    继承规则:
    - extends 值为相对路径时，相对于当前 YAML 所在目录解析
    - 子配置覆盖父配置的同名字段（浅合并）
    - 支持多级继承链（A → B → C）
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PathConfig.get().configs_dir / path

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 递归解析 extends
    extends = raw.pop("extends", None)
    if extends:
        extends_path = path.parent / extends
        parent = _load_yaml_with_inheritance(extends_path)
        merged = _deep_merge(parent, raw)
        return merged

    return raw

def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 覆盖 base"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

### 3.6 环境变量覆盖示例

```bash
# 以下命令等价于修改 YAML 中的对应字段
export CCLLM_SEED=123
export CCLLM_DTYPE=fp32
export CCLLM_TRAINING__BATCH_SIZE=16           # 双下划线表示嵌套
export CCLLM_TRAINING__LEARNING_RATE=1e-3
export CCLLM_MODEL__N_LAYERS=16

python scripts/pretrain.py --config configs/pretrain.yaml
```

### 3.7 使用示例

```python
# scripts/pretrain.py

from classic_chinese_llm.config import load_config, PretrainConfig
from classic_chinese_llm.config.paths import PathConfig

def main():
    # 1. 初始化路径（通常在入口脚本做一次）
    PathConfig.initialize(project_root=".")

    # 2. 加载配置
    config = load_config("configs/pretrain.yaml", PretrainConfig)

    # 3. 类型安全使用——IDE 自动补全
    print(f"Model: d_model={config.model.d_model}, n_layers={config.model.n_layers}")
    print(f"Training: lr={config.training.learning_rate}, batch={config.training.batch_size}")
    print(f"Dtype: {config.dtype}, Seed: {config.seed}")

    # 4. 路径使用
    paths = PathConfig.get()
    checkpoint_dir = paths.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

---

## 4. 关键技术点

### 4.1 `model_validator` 跨字段校验

Pydantic v2 的 `@model_validator(mode="after")` 在单个字段校验完成后运行，可以访问所有已校验字段。适用于 `max_steps` 与 `max_epochs` 互斥、`d_ff` 应为 `d_model` 的倍数等跨字段约束。

```python
@model_validator(mode="after")
def check_d_ff_multiple(self):
    if self.model.d_ff % self.model.d_model != 0:
        raise ValueError(f"d_ff ({self.model.d_ff}) 应为 d_model ({self.model.d_model}) 的整数倍")
    return self
```

### 4.2 `field_validator` 单字段校验

适用于单字段的值范围检查。但大多数简单检查可以用 `Field(ge=..., le=...)` 直接完成，避免写 validator 函数。

### 4.3 pydantic-settings 环境变量注入原理

`BaseSettings` 继承自 `BaseModel`，在实例化时按优先级合并多个来源：

```
命令行参数 > 环境变量 CCLLM_* > .env 文件 > YAML 文件 > Field default
```

`env_nested_delimiter="__"` 使得 `CCLLM_TRAINING__BATCH_SIZE` 映射到 `settings.training.batch_size`。

### 4.4 YAML 继承 vs Hydra 的配置组

本项目选择简单的 YAML `extends` 机制而非 Hydra 的 ConfigGroup 有两个原因：

1. **显式优于隐式**: `extends: "default.yaml"` 明确表达了继承关系
2. **不引入框架耦合**: Hydra 的 `@hydra.main` 装饰器会替换整个 `main` 函数签名，与普通 Python 脚本的行为差异较大

对于只有 3 个配置文件的项目，轻量方案足矣。

### 4.5 PathConfig 单例模式

`PathConfig` 使用类级单例而非全局变量，原因：

- **可测试**: 测试中可以 `PathConfig.initialize(tmp_path)` 重置路径
- **避免循环导入**: 不依赖任何其他模块
- **惰性解析**: 只有被访问的属性才计算路径

---

## 5. 与其他模块的关系

```
Config ─── 被依赖 ───> Utils (logging 需要 LoggingConfig)
Config ─── 被依赖 ───> Training (Trainer 接收 Settings)
Config ─── 被依赖 ───> Inference (Engine 需要 ModelConfig + 路径)
Config ─── 被依赖 ───> Data (Collector 需要路径 + seed)
```

配置系统是所有其他模块的依赖源，因此 Phase 1 中 config 模块必须最先完成。

---

## 6. 验证清单

- [ ] `load_config("pretrain.yaml", PretrainConfig)` 返回正确的 PretrainConfig 实例
- [ ] YAML 缺失必填字段时抛出清晰的 `ValidationError`
- [ ] `CCLLM_SEED=999` 环境变量覆盖 YAML 中的 seed 值
- [ ] `extends` 继承链正确深度合并嵌套字典
- [ ] `PathConfig.get()` 在未初始化时抛出 `RuntimeError`
- [ ] `mypy` 对配置字段的类型检查正常工作（故意写错类型应报错）
