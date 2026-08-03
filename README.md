# Classic Chinese LLM（文言文大语言模型）

从零实现一个 GPT-2 风格 Decoder-only Transformer（~157M 参数），仅使用 `torch.nn` 原生模块构建，在文言文语料上完成预训练和指令微调全流程，最终产出可进行文言文对话的 LLM。

## 项目定位

- **学习第一**：深入理解 Transformer 的每一个组件，从 RMSNorm、RoPE、Multi-Head Attention 到 SwiGLU FFN，全部从零手写
- **零 HF 模型代码依赖**：不使用 `transformers`、`peft`、`trl` 等库构建模型，`torch.nn` 是唯一的模型构建工具
- **单卡可训练**：~157M 参数量，混合精度训练下约需 8-9GB 显存，消费级 12GB GPU 即可运行

## 架构概览

```
┌──────────────────────────────────────────────────────┐
│              对话界面层 (Chat)                         │
│   Gradio Web UI  │  FastAPI (OpenAI 兼容)  │  SSE    │
└──────────────────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────┐
│              推理服务层 (Inference)                    │
│   模型加载 → 自回归生成 → 采样策略 → 流式输出          │
└──────────────────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────┐
│              模型层 (Model) — 核心                     │
│   RMSNorm → RoPE → MultiHeadAttention → SwiGLU FFN  │
│         TransformerBlock × 14 (Pre-norm)             │
└──────────────────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────┐
│              训练层 (Training)                         │
│   Trainer │ Pretrain │ SFT │ Callbacks │ DataCollator│
└──────────────────────────────────────────────────────┘
                          │
┌────────────────────┬─────────────────────────────────┐
│   Tokenizer 层      │        数据层 (Data)            │
│ SentencePiece       │ 采集 → 清洗 → 去重 → 格式化     │
│ Unigram 32K vocab   │ 多源适配器模式                   │
└────────────────────┴─────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────┐
│              基础设施层 (Infra)                         │
│ Pydantic+YAML 配置 │ Rich 日志 │ 设备检测 │ Checkpoint │
└──────────────────────────────────────────────────────┘
```

### 模型规格

| 参数 | 数值 |
|------|------|
| 词表大小 (vocab_size) | 32,000 |
| 隐藏维度 (d_model) | 768 |
| 层数 (n_layers) | 14 |
| 注意力头数 (n_heads) | 12 |
| 前馈维度 (d_ff) | 3,072 |
| 最大序列长度 (max_seq_len) | 2,048 |
| 总参数量 | ~157M |
| Embedding / LM Head | 权重共享 |

核心架构选择：**RoPE** 位置编码、**SwiGLU** 激活函数、**RMSNorm** 归一化、**Pre-norm** 残差连接、**FlashAttention**（通过 `F.scaled_dot_product_attention`）。

详细设计见 [架构设计文档](docs/architecture.md)，各模块的深入设计见 [docs/design/](docs/design/) 目录。

## 环境配置

### 硬件要求

| 阶段 | 最低显存 | 推荐显存 | 预计耗时 |
|------|---------|---------|---------|
| 数据处理 | CPU 即可 | 16GB RAM | 1-2 小时 |
| Tokenizer 训练 | CPU 即可 | 16GB RAM | 10-30 分钟 |
| 预训练 | 8GB VRAM | 12GB+ VRAM | 2-3 天 |
| 指令微调 | 8GB VRAM | 12GB+ VRAM | 2-4 小时 |
| 推理 | 4GB VRAM | 8GB+ VRAM | 实时 |

### Conda 环境搭建

```bash
# 1. 创建 Python 3.12 虚拟环境
conda create -n classic-llm python=3.12 -y
conda activate classic-llm

# 2. 安装 PyTorch（CUDA 12.4 版本，根据你的 CUDA 版本调整）
#    前往 https://pytorch.org/get-started/locally/ 查看适合你的安装命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 安装项目依赖（包含数据处理、对话服务、开发工具等全部依赖）
pip install -r requirements.txt

# 4. 注册项目模块（本项目使用 src 布局，二选一）
#    方式 A：可编辑安装（推荐，一次配置永久生效）
pip install -e .
#    方式 B：设置 PYTHONPATH（免安装，每次打开终端需重新设置）
#    Linux/macOS: export PYTHONPATH="$(pwd)/src"
#    Windows PowerShell: $env:PYTHONPATH = "$(Get-Location)\src"

# 5. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from classic_chinese_llm import __version__; print(f'classic_chinese_llm {__version__} 加载成功')"
```

### 依赖说明

全部依赖已集中写在 `requirements.txt` 中，按功能分组并标注了版本号。

```bash
# 一次性安装全部依赖（推荐）
pip install -r requirements.txt
```

`requirements.txt` 中已包含以下四组依赖。如果你不需要某组（比如不做数据采集），可以编辑文件注释掉对应部分：

| 分组 | 主要依赖 | 用途 |
|------|---------|------|
| 核心 | torch, datasets, accelerate, sentencepiece, pydantic 等 | 模型构建、训练、配置 |
| 数据处理 | requests, beautifulsoup4, datasketch 等 | 文言文语料采集、清洗、去重 |
| 对话服务 | gradio, fastapi, uvicorn, sse-starlette | Web UI + REST API |
| 开发工具 | black, ruff, mypy, pytest | 代码格式化、静态检查、测试 |

### CUDA 版本兼容

如果遇到 CUDA 版本不匹配，可以通过环境变量指定：

```bash
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"

# Linux / macOS
export CUDA_VISIBLE_DEVICES=0
```

## 快速开始

完整流程共五步：准备语料 → 采集处理 → 训练分词器 → 预训练 → 指令微调 → 对话。

### 第零步：下载原始语料（手动）

`collect_data.py` **不会自动下载数据**——你需要先将原始文件放到 `data/raw/<source>/` 下。建议优先下载殆知阁（占数据量 80%+），其他按需补充。

| 数据源 | 目录 | 获取方式 |
|--------|------|---------|
| **殆知阁** ⭐ | `data/raw/daizhige/` | `git clone --depth 1 https://github.com/garychowcmu/daizhigev20.git data/raw/daizhige` |
| GitHub 语料 | `data/raw/github/` | 如 [NiuTrans/Classical-Modern](https://github.com/NiuTrans/Classical-Modern)（文言-白话平行语料，可选） |
| 四库全书 | `data/raw/siku/` | [Project Gutenberg #7221](https://www.gutenberg.org/ebooks/7221)（公共领域子集，可选） |
| ctext.org | `data/raw/ctext/` | [ctext.org](https://ctext.org/zh) 注册后逐章手动下载（质量最高，精选几本即可） |

> 详细下载指南与备选镜像见 [docs/guide.md](docs/guide.md#前置准备下载原始语料)。

```bash
# 确保已激活 conda 环境
conda activate classic-llm

# 第一步：采集并处理文言文语料
python scripts/collect_data.py --raw-dir data/raw

# 第二步：训练 SentencePiece Unigram 分词器
python scripts/train_tokenizer.py --corpus data/processed/deduplicated.jsonl --vocab-size 32000

# 第三步：预训练（~2-3 天，建议使用 tmux/screen 后台运行）
python scripts/pretrain.py --config configs/pretrain.yaml

# 第四步：指令微调（~2-4 小时）
python scripts/finetune.py \
    --config configs/sft.yaml \
    --pretrained-checkpoint models/checkpoints/checkpoint_best.pt

# 第五步：启动对话
# Gradio Web UI（浏览器中打开 http://localhost:7860）
python scripts/chat.py --checkpoint models/checkpoints/sft_best.pt

# 或启动 API 服务（OpenAI 兼容格式）
python scripts/serve.py --checkpoint models/checkpoints/sft_best.pt --port 8000
```

> **提示**：如果只需要某个环节（比如已有 checkpoint 只想推理），请查看 [docs/guide.md](docs/guide.md) 按场景跳转。

## 项目结构

```
classic-chinese-llm/
├── configs/                         # YAML 配置文件
│   ├── default.yaml                 #   基础默认配置
│   ├── pretrain.yaml                #   预训练配置
│   ├── sft.yaml                     #   指令微调配置
│   └── eval.yaml                    #   评测配置
│
├── src/classic_chinese_llm/
│   ├── config/                      # Pydantic 配置系统 + 路径管理
│   ├── data/                        # 数据管道
│   │   ├── sources/                 #   可插拔数据源适配器
│   │   │   ├── daizhige.py          #     殆知阁
│   │   │   ├── github_corpora.py    #     GitHub 开源语料
│   │   │   ├── sikuquanshu.py       #     四库全书公开子集
│   │   │   └── ctext.py             #     ctext.org
│   │   ├── collector.py             #   数据采集编排
│   │   ├── cleaner.py               #   清洗（规范化、去噪）
│   │   ├── deduplicator.py          #   去重（SHA-256 + MinHash LSH）
│   │   └── formatter.py             #   指令数据集构建
│   ├── tokenizer/                   # SentencePiece Unigram 分词器
│   ├── model/                       # Transformer 核心实现
│   │   ├── layers.py                #   RMSNorm、RoPE、Attention、SwiGLU FFN
│   │   ├── transformer.py           #   TransformerBlock + TransformerLM
│   │   └── generation.py            #   自回归生成 + 采样策略
│   ├── training/                    # 训练框架
│   │   ├── trainer.py               #   通用训练循环
│   │   ├── pretrain.py              #   预训练 Runner
│   │   ├── sft.py                   #   指令微调 Runner
│   │   ├── callbacks.py             #   回调插件
│   │   └── data_collator.py         #   动态 Padding + Label Masking
│   ├── evaluation/                  # 评估与评测
│   ├── inference/                   # 推理引擎
│   ├── chat/                        # 对话界面 (Gradio + FastAPI)
│   └── utils/                       # 日志、设备检测、Checkpoint
│
├── scripts/                         # CLI 入口脚本
│   ├── collect_data.py
│   ├── train_tokenizer.py
│   ├── pretrain.py
│   ├── finetune.py
│   ├── evaluate.py
│   ├── chat.py
│   └── serve.py
│
├── tests/                           # 测试（与 src/ 结构一致）
├── docs/                            # 文档
│   ├── architecture.md              #   架构设计文档
│   ├── guide.md                     #   场景使用指南
│   └── design/                      #   模块详细设计文档
│       ├── 09-model.md              #     模型层设计
│       ├── 08-tokenizer.md          #     Tokenizer 设计
│       ├── 10-training.md           #     训练层设计
│       └── ...                      #     等共 13 份设计文档
│
├── requirements.txt                 # 依赖快照（由 pyproject.toml 导出）
├── pyproject.toml                   # 项目元数据 + 依赖权威定义
└── CLAUDE.md                        # Claude Code 开发指引
```

## 配置系统

配置基于 Pydantic + YAML，支持三层合并：`default.yaml` → 阶段配置 → 环境变量覆盖。

```bash
# 使用环境变量覆盖任意配置项（前缀 CCLLM_，嵌套用 __ 分隔）
CCLLM_TRAINING__BATCH_SIZE=16 CCLLM_TRAINING__LEARNING_RATE=1e-4 \
    python scripts/pretrain.py --config configs/pretrain.yaml

# 从 checkpoint 恢复训练
python scripts/pretrain.py --config configs/pretrain.yaml --resume
```

## 开发指南

```bash
# 确保已安装全部依赖（含开发工具）
pip install -r requirements.txt

# 代码格式化
black src/ tests/

# 静态检查
ruff check src/ tests/
mypy src/

# 运行测试
pytest tests/ -v
pytest tests/test_model/test_layers.py -v   # 单个测试文件
```

## 技术栈

| 类别 | 依赖 | 用途 |
|------|------|------|
| 深度学习 | PyTorch ≥2.4 | 模型构建、自动微分、FlashAttention |
| 数据处理 | datasets, accelerate | 数据加载、混合精度训练 |
| 分词 | sentencepiece, tokenizers | SentencePiece 训练 + HF 封装 |
| 配置 | pydantic, pyyaml | 类型安全配置管理 |
| 对话 | gradio, fastapi, uvicorn | Web UI + REST API |
| 质量 | black, ruff, mypy, pytest | 格式化、Lint、类型检查、测试 |

> **重要约束**：`transformers` 库仅用于 tokenizer 互操作，不参与模型构建。所有模型代码仅依赖 `torch.nn`。

## 许可证

MIT
