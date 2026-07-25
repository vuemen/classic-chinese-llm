# 项目基础设施设计文档

**所属阶段:** Phase 1 — 基础设施
**涉及模块:** 项目骨架、pyproject.toml、CI 配置
**日期:** 2026-07-25

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | 项目骨架 | 按 `src/` layout 建立完整目录树，含所有 Phase 模块的 `__init__.py` 占位 |
| F2 | 依赖管理 | `pyproject.toml` 声明核心依赖、可选依赖分组（data/chat/dev），支持 `pip install -e` 可编辑安装 |
| F3 | 代码质量 CI | black 格式化 + ruff lint + mypy 类型检查，本地可执行，CI 门禁强制通过 |
| F4 | 配置骨架 | `configs/` 目录下 default.yaml、pretrain.yaml、sft.yaml 配置文件模板 |
| F5 | 测试骨架 | `tests/` 目录镜像 `src/` 结构，包含 `conftest.py` 和基础测试 |

### 1.2 非功能需求

- **Python 版本**: ≥3.12, <3.14（CLAUDE.md 约束）
- **零 HF 模型代码依赖**: `transformers` 仅用于 tokenizer 互操作，模型构建只用 `torch.nn`
- **单仓库结构**: 所有代码在 `src/classic_chinese_llm/` 下，脚本入口在 `scripts/`

---

## 2. 方案选型与对比

### 2.1 构建系统

Python 项目构建系统的核心选择：

| 方案 | 优势 | 劣势 | 结论 |
|------|------|------|------|
| **setuptools** | PyTorch 生态事实标准；`pip install -e` 直接可用；社区最广泛支持 | `setup.cfg`/`setup.py` 历史包袱，需额外配置 | ✅ 选用 |
| hatch | 现代化，PEP 621 原生支持，插件体系好 | PyTorch 社区采用率低，CI 中需额外安装 hatch | ❌ |
| poetry | 依赖解析最好，lock 文件完善 | 与 PyTorch CUDA 版本管理常有摩擦，构建速度慢 | ❌ |
| pdm | PEP 621 原生，类似 npm 体验 | 国内生态认知度低，团队门槛高 | ❌ |

**最终选择**: **setuptools** + PEP 621 `pyproject.toml` 声明元数据。这是 PyTorch 项目的主流选择，与后续的 `torch`、`datasets`、`accelerate` 等依赖兼容性最好。

### 2.2 Linting & Formatting 工具链

| 工具组合 | 优势 | 劣势 | 结论 |
|----------|------|------|------|
| **black + ruff + mypy** | ruff 极快（Rust 实现），统一替代 flake8/isort/pyflakes；black 零配置 | ruff 对部分小众规则支持不如 flake8 插件完整 | ✅ 选用 |
| black + flake8 + isort + mypy | 成熟稳定，插件丰富 | 配置散落在多处，速度慢（Python 实现） | ❌ |
| ruff（all-in-one）| 一个工具全覆盖 | ruff 的 formatter 尚未完全稳定（截至 2024），用 black 更稳妥 | ❌ 暂不采用 |

**最终选择**: **black + ruff + mypy**。ruff 已足够成熟用于 lint 和 import 排序，black 作为 formatter 更稳妥。这套组合也是 CLAUDE.md 的强制要求。

### 2.3 CI 运行方式

| 方式 | 优势 | 劣势 | 结论 |
|------|------|------|------|
| pre-commit hooks | 提交前自动检查，零 CI 消耗 | 依赖开发者本地安装，可被跳过 | 作为本地辅助 |
| GitHub Actions | 门禁强制，不依赖本地环境 | 需 GitHub 仓库，有延迟 | 作为正式门禁 |
| 手动命令行 | 灵活，无需额外配置 | 纯靠纪律，容易遗漏 | 作为日常开发 |

**最终选择**: **日常开发手动命令 + GitHub Actions CI 门禁**。`.pre-commit-config.yaml` 已列入 `.gitignore`，不强制但可供选用。

---

## 3. 最终方案

### 3.1 目录结构

```
classic-chinese-llm/
├── pyproject.toml                  # PEP 621 项目元数据 + 工具配置
├── README.md
├── LICENSE                         # MIT
├── CLAUDE.md                       # Claude Code 项目指引
├── .gitignore
│
├── docs/
│   ├── architecture.md             # 架构全景文档
│   └── design/                     # 模块设计文档（当前文档系列）
│       ├── 01-project-infrastructure.md
│       ├── 02-config-system.md
│       └── 03-utils-module.md
│
├── configs/
│   ├── default.yaml                # 通用默认配置
│   ├── pretrain.yaml               # 预训练配置
│   └── sft.yaml                    # 指令微调配置
│
├── src/
│   └── classic_chinese_llm/
│       ├── __init__.py             # 包版本号
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py         # Pydantic Settings 模型
│       │   └── paths.py            # PathConfig 路径常量
│       ├── data/
│       │   ├── __init__.py
│       │   └── sources/
│       │       └── __init__.py
│       ├── tokenizer/
│       │   └── __init__.py
│       ├── model/
│       │   └── __init__.py
│       ├── training/
│       │   └── __init__.py
│       ├── evaluation/
│       │   └── __init__.py
│       ├── inference/
│       │   └── __init__.py
│       ├── chat/
│       │   └── __init__.py
│       └── utils/
│           └── __init__.py
│
├── scripts/
│   ├── __init__.py                 # 使 scripts 可作为子包导入
│   ├── collect_data.py
│   ├── train_tokenizer.py
│   ├── pretrain.py
│   ├── finetune.py
│   ├── evaluate.py
│   ├── chat.py
│   └── serve.py
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_config/
    │   └── __init__.py
    ├── test_data/
    │   └── __init__.py
    ├── test_tokenizer/
    │   └── __init__.py
    ├── test_model/
    │   └── __init__.py
    └── test_training/
        └── __init__.py
```

### 3.2 pyproject.toml 设计

```toml
[build-system]
requires = ["setuptools>=75.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "classic-chinese-llm"
version = "0.1.0"
description = "Classical Chinese conversational LLM — a ~157M Decoder-only Transformer built from scratch"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.12,<3.14"

dependencies = [
    "torch>=2.4.0",
    "datasets>=2.21.0",
    "accelerate>=0.34.0",
    "sentencepiece>=0.2.0",
    "tokenizers>=0.20.0",
    "transformers>=4.45.0",       # 仅 tokenizer 互操作
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
    "pyyaml>=6.0",
    "rich>=13.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
data = [
    "requests>=2.32",
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "datasketch>=1.6",
]
chat = [
    "gradio>=5.0",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "sse-starlette>=2.1",
]
dev = [
    "black>=24.0",
    "ruff>=0.7",
    "mypy>=1.12",
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[project.scripts]
classic-llm-collect = "scripts.collect_data:main"
classic-llm-train-tokenizer = "scripts.train_tokenizer:main"
classic-llm-pretrain = "scripts.pretrain:main"
classic-llm-finetune = "scripts.finetune:main"
classic-llm-chat = "scripts.chat:main"
classic-llm-serve = "scripts.serve:main"

[tool.setuptools]
packages = {find = {where = ["src"]}}

[tool.black]
line-length = 100
target-version = ["py312"]
include = '\\.pyi?$'
extend-exclude = '''
/(
  | \.eggs
  | \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = [
    "E", "W", "F", "I",    # pycodestyle + pyflakes + isort
    "N",                     # pep8-naming
    "B",                     # flake8-bugbear
    "SIM",                   # flake8-simplify
    "C4",                    # flake8-comprehensions
    "UP",                    # pyupgrade
]
ignore = [
    "E501",                  # 行长度由 black 处理
]

[tool.ruff.lint.isort]
known-first-party = ["classic_chinese_llm"]

[tool.mypy]
python_version = "3.12"
strict = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_return_any = true
warn_unused_ignores = true
exclude = "build|dist|\.venv|\.tox"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short --strict-markers"
```

### 3.3 配置文件模板

**configs/default.yaml**:

```yaml
# 通用默认配置 — 被 pretrain.yaml / sft.yaml 继承和覆盖
seed: 42
dtype: "bf16"               # bf16 | fp16 | fp32

logging:
  level: "INFO"
  log_dir: "logs"

paths:
  data_dir: "data"
  model_dir: "models"
  checkpoint_dir: "models/checkpoints"

model:
  vocab_size: 32000
  d_model: 768
  n_layers: 14
  n_heads: 12
  d_ff: 3072
  max_seq_len: 2048
  dropout: 0.0
```

**configs/pretrain.yaml**:

```yaml
# 继承 default.yaml
extends: "default.yaml"

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  weight_decay: 0.1
  warmup_steps: 1000
  max_steps: 100000
  eval_every: 500
  save_every: 2000
  max_checkpoints: 5

optimizer:
  name: "adamw"
  betas: [0.9, 0.95]
  eps: 1.0e-8

scheduler:
  name: "cosine"
  min_lr: 3.0e-5
```

**configs/sft.yaml**:

```yaml
# 继承 default.yaml
extends: "default.yaml"

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 100
  max_epochs: 3
  eval_every: 200
  save_every: 500

chat_template: "classical_chinese_v1"

data:
  max_samples: 15000
  val_split: 0.05
```

### 3.4 CI 工作流（可选）

```yaml
# .github/workflows/ci.yml（不入库，按需创建）
name: CI
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install ".[dev]"
      - run: black --check src/ tests/
      - run: ruff check src/ tests/
      - run: mypy src/
```

---

## 4. 关键技术点

### 4.1 PEP 621 `[project]` 声明

PEP 621 将项目元数据标准化到 `pyproject.toml` 的 `[project]` 节，不再需要 `setup.cfg` 或 `setup.py`。setuptools 从 v61.0 起完全支持，`pip install -e .` 直接可用。

### 4.2 `src/` Layout 优势

使用 `src/` 布局（代码在 `src/classic_chinese_llm/`）而非 flat layout（代码在根目录 `classic_chinese_llm/`）的原因：

- **防止意外导入**: `pip install -e .` 后导入的是已安装的包，而非当前目录，避免了路径混淆
- **强制可安装性**: 代码必须通过 `pip install` 才能使用，不会出现"在我机器上能跑"的问题
- **测试视角**: 测试代码通过 `import classic_chinese_llm` 导入，与最终用户视角一致

### 4.3 ruff 统一 lint 栈

ruff 在单个 Rust 二进制中实现了以下工具的全部规则：

| 原工具 | ruff 规则前缀 |
|--------|-------------|
| flake8 (pycodestyle) | E, W |
| pyflakes | F |
| isort | I |
| flake8-bugbear | B |
| pyupgrade | UP |

传统方案需要 pip 安装 5+ 工具 + 协调各自配置，ruff 一个 `pyproject.toml` 节搞定。

### 4.4 mypy strict 模式

`strict = true` 启用全部严格检查选项。这意味着：
- 所有函数必须有完整类型注解
- 不允许 `Any` 隐式扩散
- 不允许未类型化的装饰器

这是 CLAUDE.md 强制要求的"所有函数签名必须包含完整类型注解"的技术保证。

### 4.5 配置继承机制

YAML 配置文件间通过 `extends` 字段实现继承：加载 `pretrain.yaml` 时自动递归加载 `default.yaml` 并深度合并。这避免了在多个配置文件中重复通用参数。实现细节见 [02-config-system.md](02-config-system.md)。

---

## 5. 验证清单

- [ ] `pip install -e ".[dev]"` 成功，所有依赖版本兼容
- [ ] `black --check src/ tests/` 通过（空目录视为通过）
- [ ] `ruff check src/ tests/` 通过
- [ ] `mypy src/` 通过
- [ ] `pytest tests/ -v` 通过（至少 `conftest.py` 可加载）
- [ ] 所有 `__init__.py` 存在，目录树与架构文档一致
