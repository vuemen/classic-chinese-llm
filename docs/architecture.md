# Classical Chinese LLM 项目架构文档

**项目名称:** classic-chinese-llm
**Python 版本:** 3.12
**硬件要求:** NVIDIA GPU（12GB+ VRAM），BF16 混合精度训练
**日期:** 2026-07-25

---

## 1. 项目定位

从零实现一个 GPT-2 风格 Decoder-only Transformer（~157M 参数），使用文言文语料完成预训练和指令微调全流程，最终产出可进行文言文对话的 LLM。

核心约束：
- **零 HF 模型代码依赖**：仅使用 `torch.nn` 原生模块构建 Transformer，HuggingFace 生态（`datasets`、`tokenizers`、`accelerate`）只用于数据加载和训练加速
- **单卡可训练**：参数规模和训练配置针对消费级 GPU（12GB VRAM）优化
- **数据规模现实**：文言文可用语料总量约 3-6 亿字符（2-4 亿 token），在此约束下设计模型规模

> 📖 使用指南请见 [guide.md](guide.md)，各模块详细设计见 [design/](design/) 目录。

---

## 2. 架构全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                        对话界面层 (Chat)                          │
│   Gradio Web UI  │  FastAPI REST API (OpenAI 兼容)  │  SSE 流式   │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                       推理服务层 (Inference)                       │
│        模型加载  │  自回归生成  │  采样策略  │  流式输出             │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                      模型层 (Model) — 核心                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ RMSNorm  │  │   RoPE   │  │ MultiHead│  │  SwiGLU FFN  │    │
│  │ 归一化    │  │ 位置编码  │  │ Attention│  │  前馈网络     │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
│                     TransformerBlock × N                         │
│                  Pre-norm 残差连接 + FlashAttention               │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                      训练层 (Training)                            │
│    Trainer (训练循环)  │  Pretrain  │  SFT  │  Callbacks          │
│    梯度累积 + 混合精度 + Checkpoint 管理                          │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────┬──────────────────────────────────────┐
│     Tokenizer 层          │           数据层 (Data)               │
│  SentencePiece Unigram    │  数据源适配 → 清洗 → 去重 → 格式化    │
│  文言文专用 32K vocab     │  原始文本 → 指令数据集                 │
└──────────────────────────┴──────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                      基础设施层 (Infra)                            │
│  配置管理 (Pydantic + YAML)  │  日志  │  设备检测  │  Checkpoint   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 分层定义

### 3.1 基础设施层

**职责**：为所有上层模块提供配置、日志、设备管理和模型持久化能力。

| 组件 | 职责 |
|------|------|
| 配置管理 | 基于 Pydantic + YAML 的类型安全配置系统。所有模块的参数均通过配置对象注入，支持环境变量覆盖。分为默认配置、预训练配置、SFT 配置 |
| 日志系统 | 基于 `logging` + `rich` 的结构化日志，同时输出到控制台和文件 |
| 设备检测 | GPU 可用性检测、显存报告、混合精度兼容性检查 |
| Checkpoint | 模型权重 + 优化器状态 + 训练元信息的保存与恢复 |

> 📖 详细设计：[01-project-infrastructure](design/01-project-infrastructure.md)、[02-config-system](design/02-config-system.md)、[03-utils-module](design/03-utils-module.md)

### 3.2 数据层

**职责**：文言文语料的采集、清洗、去重，以及预训练/指令微调数据集的构建。

**数据流**：
```
[数据源文件] → Collector → Cleaner → Deduplicator → ┬→ 预训练数据集
                                                     └→ Formatter → 指令微调数据集
```

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| Collector | 编排多个数据源，统一的发现→解析→校验流程 | 各数据源的原始文件（txt/jsonl/xml） | 统一格式的原始 JSONL |
| Cleaner | Unicode 规范化、现代标点剥离、版式噪声去除、长度过滤 | 原始 JSONL | 清洗后 JSONL |
| Deduplicator | SHA-256 精确去重 + MinHash+LSH 近似去重（Jaccard ≥0.85） | 清洗后 JSONL | 去重后 JSONL |
| Formatter | 将原始文言文段落通过模板转换为指令-响应对 | 去重后 JSONL | 指令数据集 JSONL |

**数据源**（设计为可插拔的适配器模式）：

| 来源 | 类型 | 说明 |
|------|------|------|
| 殆知阁 | 主力 | 古代汉语语料库，可直接下载 txt 打包文件 |
| GitHub 开源语料 | 主力 | 社区维护的文言文合集 |
| 四库全书公开子集 | 补充 | 部分可在开源镜像站获取 |
| ctext.org | 手动补充 | 质量最高但无批量接口，仅用于补全缺失典籍 |

每个数据源通过实现统一适配器接口接入，新增来源不影响现有流程。

> 📖 详细设计：[04-data-collector](design/04-data-collector.md)、[05-data-cleaner](design/05-data-cleaner.md)、[06-data-deduplicator](design/06-data-deduplicator.md)、[07-data-formatter](design/07-data-formatter.md)

### 3.3 Tokenizer 层

**职责**：为文言文训练专用的子词分词器。

**设计决策**：
- 选用 **SentencePiece Unigram** 模型，vocab size = 32,000
- 训练数据仅来源于清洗后的文言文语料，不混用现代文本
- `character_coverage = 0.99995`，覆盖几乎全部文言字符
- `byte_fallback = True`，零 OOV 保证
- 训练后通过 HF `tokenizers` 封装为标准 `PreTrainedTokenizerFast`，与 `datasets` 库无缝互操作

> 📖 详细设计：[08-tokenizer](design/08-tokenizer.md)

**模块结构**：

| 组件 | 职责 |
|------|------|
| Tokenizer Trainer | SentencePiece 训练封装，参数配置，模型导出 |
| Pre-tokenizer | 文言文专用预分词规则（按句读标点断句，保留字符原貌） |

### 3.4 模型层（核心）

**职责**：Decoder-only Transformer 的纯 `torch.nn` 实现，这是整个项目最主要的学习目标。

**不依赖任何 HF 模型代码**，每个组件从零实现。

**架构组成**（自底向上）：

| 组件 | 说明 | 关键设计点 |
|------|------|-----------|
| RoPE | 旋转位置编码 | 直接应用到 Q/K 向量，支持训练时未见的序列长度 |
| RMSNorm | 归一化层 | 去掉平移参数仅保留缩放，比 LayerNorm 更快 |
| MultiHeadAttention | 多头因果注意力 | 通过 `F.scaled_dot_product_attention` 后端自动使用 FlashAttention；causal mask 保证自回归 |
| SwiGLU FFN | 前馈网络 | gate + up projection → SiLU → element-wise multiply → down projection |
| TransformerBlock | 单个 Transformer 层 | Pre-norm 结构：RMSNorm → Attention → Residual → RMSNorm → FFN → Residual |
| TransformerLM | 完整模型 | Token Embedding → TransformerBlock × N → Final RMSNorm → LM Head |

**模型规格**：

| 参数 | 数值 |
|------|------|
| vocab_size | 32,000 |
| d_model | 768 |
| n_layers | 14 |
| n_heads | 12 |
| d_ff | 3,072 |
| max_seq_len | 2,048 |
| 总参数量 | ~157M |
| Embedding / LM Head | 权重共享，节省 ~24M 参数 |

**生成模块**：独立的 `Generator` 类，支持 Greedy / Temperature / Top-K / Top-P / Repetition Penalty / Beam Search，以及逐 token 流式输出。

> 📖 详细设计：[09-model](design/09-model.md) — 包含完整架构图、方案选型对比（RoPE vs Learned、SwiGLU vs ReLU、Pre-norm vs Post-norm 等）、参数量逐项计算、显存估算

### 3.5 训练层

**职责**：提供通用训练框架及预训练/指令微调两个阶段的训练逻辑。

**模块结构**：

| 组件 | 职责 |
|------|------|
| Trainer | 通用训练循环：梯度累积、自动混合精度（BF16）、学习率调度、梯度裁剪。checkpoint 保存/恢复支持中断续训 |
| Pretrain | Causal LM 预训练：对全序列计算 next-token cross-entropy loss |
| SFT | 指令微调：Chat template 格式化，loss 仅计算在 assistant 回复 token 上（label masking） |
| Callbacks | 插件式回调：LoggingCallback、CheckpointCallback、EarlyStoppingCallback |
| Data Collator | 动态 batch 内 padding、attention mask 构建、SFT 的 label masking |

**训练阶段**：

| 阶段 | 数据 | Loss 策略 | 预计耗时 (12GB GPU) |
|------|------|-----------|---------------------|
| 预训练 | 清洗后的文言文原始文本 | 全序列 Causal LM | 2-3 天 |
| 指令微调 | 5K-15K 指令-响应对 | 仅计算回复 token 的 loss | 2-4 小时 |

**12GB VRAM 可行性**：混合精度训练中每参数占用 16 bytes（BF16 权重 2B + BF16 梯度 2B + FP32 Master Weight 4B + Adam m 4B + Adam v 4B），157M 参数合计约 2.5GB，加上激活值和 CUDA 开销共约 8-9GB，在 12GB 范围内安全运行。

> 📖 详细设计：[10-training](design/10-training.md) — 包含 Trainer 设计、预训练/SFT 流程、回调系统、Data Collator、显存预算分析

### 3.6 推理服务层

**职责**：将训练好的模型加载并提供统一的生成接口。

| 组件 | 职责 |
|------|------|
| Inference Engine | 加载 checkpoint → 实例化模型 → 提供 `generate()` 和 `stream()` 接口 |

该层屏蔽了底层模型实现细节，上层对话界面只与 Inference Engine 交互。

> 📖 详细设计：[12-inference](design/12-inference.md)

### 3.7 对话界面层

**职责**：提供人机交互界面和 API 服务。

| 组件 | 技术 | 职责 |
|------|------|------|
| Web Chat UI | Gradio Blocks | 文言文风格对话界面，参数可调（temperature、top-p 等），系统提示词可切换 |
| REST API | FastAPI | `POST /v1/chat/completions`，OpenAI 兼容格式，支持 SSE 流式响应 |
| 对话管理 | 内存存储 | 多轮对话历史维护，自动截断超出上下文限制的历史消息 |
| 系统提示词 | 模板集合 | 预设角色：古文专家、诗词创作、历史讲述、文言翻译 |

> 📖 详细设计：[13-chat](design/13-chat.md)

### 3.8 评估与评测层

**职责**：评估模型在文言文任务上的表现。

| 组件 | 职责 |
|------|------|
| Perplexity | 在 held-out 测试集上计算困惑度 |
| NLG 指标 | BLEU-4、ROUGE-L、字符级准确率 |
| LLM-as-Judge | 使用评估准则对生成结果进行质量打分 |
| 评测报告 | 结构化评测报告（控制台 + JSON） |

> 📖 详细设计：[11-evaluation](design/11-evaluation.md)

---

## 4. 技术栈

### 核心

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥3.12,<3.14 | 运行时 |
| PyTorch | ≥2.4.0 | 模型构建 + 自动微分 + FlashAttention (sdpa) |
| datasets | ≥2.21.0 | 数据加载、缓存、切分 |
| accelerate | ≥0.34.0 | 混合精度训练、设备管理 |
| sentencepiece | ≥0.2.0 | Tokenizer 训练 |
| tokenizers | ≥0.20.0 | HF 快速 Tokenizer（封装 SentencePiece） |
| transformers | ≥4.45.0 | 仅用于 tokenizer 互操作，不参与模型构建 |

### 数据处理

| 依赖 | 版本 | 用途 |
|------|------|------|
| pydantic | ≥2.9 | 类型安全配置管理 |
| pyyaml | ≥6.0 | YAML 配置文件解析 |
| requests | ≥2.32 | HTTP 请求 |
| beautifulsoup4, lxml | — | HTML/XML 解析 |
| datasketch | ≥1.6 | MinHash + LSH 去重 |
| rich, tqdm | — | 终端输出与进度显示 |

### 对话服务

| 依赖 | 版本 | 用途 |
|------|------|------|
| gradio | ≥5.0 | Web 聊天界面 |
| fastapi, uvicorn | — | REST API 服务 |
| sse-starlette | ≥2.1 | SSE 流式输出 |

### 开发工具

| 依赖 | 版本 | 用途 |
|------|------|------|
| black | ≥24.0 | 代码格式化 |
| ruff | ≥0.7 | Lint + import 排序 |
| mypy | ≥1.12 | 静态类型检查 |
| pytest, pytest-cov | — | 测试与覆盖率 |

---

## 5. 模型规模与数据约束

### 5.1 为什么选择 ~157M 参数

这是 Chinchilla 最优训练法则（tokens ≈ 20 × params）与文言文数据天花板之间的折中：

| 参数量 | 最优数据需求 | 可用文言文数据 | 是否匹配 |
|--------|-------------|---------------|----------|
| 50M | ~1B token | 2-4 亿 token | 数据偏少，需大量循环 |
| **~157M** | **~3.1B token** | **2-4 亿 token** | 需约 8 轮循环，可接受 |
| 300M | ~6B token | 2-4 亿 token | 数据严重不足，极易过拟合 |

> 文言文的客观约束：高质量文本总量约 3-6 亿字符（token 化后约 2-4 亿 token）。在设计上接受这个上限，而非为填补数据而去爬取版权不明的资源。

### 5.2 12GB VRAM 运行可行性

混合精度训练（BF16 + Adam），每参数总计占用 16 bytes。157M 参数在 batch_size=8、seq_len=1024 时，包括激活值和 CUDA 开销在内总计约 8-9GB，12GB 显存可安全运行。

---

## 6. 项目目录结构

```
classic-chinese-llm/
├── pyproject.toml
├── CLAUDE.md
├── README.md
│
├── docs/
│   ├── architecture.md             # 本文档
│   ├── guide.md                    # 场景使用指南
│   └── design/                     # 模块详细设计文档（共 13 份）
│       ├── 01-project-infrastructure.md
│       ├── 02-config-system.md
│       ├── 03-utils-module.md
│       ├── 04-data-collector.md
│       ├── 05-data-cleaner.md
│       ├── 06-data-deduplicator.md
│       ├── 07-data-formatter.md
│       ├── 08-tokenizer.md
│       ├── 09-model.md
│       ├── 10-training.md
│       ├── 11-evaluation.md
│       ├── 12-inference.md
│       └── 13-chat.md
│
├── configs/
│   ├── default.yaml
│   ├── pretrain.yaml
│   ├── sft.yaml
│   └── eval.yaml
│
├── src/
│   └── classic_chinese_llm/
│       ├── __init__.py
│       ├── config/                 # 配置管理
│       ├── data/                   # 数据管道
│       │   └── sources/            #   可插拔数据源适配器
│       ├── tokenizer/              # Tokenizer 训练
│       ├── model/                  # Transformer 核心实现
│       ├── training/               # 训练框架
│       ├── evaluation/             # 评估指标与评测
│       ├── inference/              # 推理封装
│       ├── chat/                   # 对话界面
│       └── utils/                  # 通用工具
│
├── scripts/                        # CLI 入口
│   ├── collect_data.py
│   ├── train_tokenizer.py
│   ├── pretrain.py
│   ├── finetune.py
│   ├── evaluate.py
│   ├── chat.py
│   └── serve.py
│
└── tests/                          # 与 src/ 结构一致
    ├── conftest.py
    ├── test_chat/
    ├── test_config/
    ├── test_data/
    ├── test_evaluation/
    ├── test_inference/
    ├── test_model/
    ├── test_tokenizer/
    ├── test_training/
    └── test_utils/
```

---

## 7. 实施路线图

```
Phase 1 (第1-2周): 基础设施
    项目骨架、pyproject.toml、配置系统、工具模块、CI (black/ruff/mypy)

Phase 2 (第2-3周): 数据管道
    数据源适配器、采集/清洗/去重流水线、数据统计

Phase 3 (第3-4周): Tokenizer
    SentencePiece Unigram 训练、预分词规则、HF tokenizer 封装

Phase 4 (第4-6周): 模型实现与预训练 ← 核心阶段
    从零实现 RMSNorm → RoPE → Attention → SwiGLU → TransformerBlock → TransformerLM
    Trainer 通用框架 → Causal LM 预训练（~2-3 天）

Phase 5 (第6-7周): 指令微调
    指令数据集构建 → SFT 训练（~2-4 小时）→ LLM-as-Judge 评测

Phase 6 (第7-8周): 对话界面
    Gradio UI + FastAPI + 对话管理

Phase 7 (第8周+): 打磨
    测试覆盖、评估报告、文档完善
```

**依赖链**：Phase 1 → 2 → 3 → 4 → 5 → 6 → 7（严格串行，每阶段输出是下一阶段的输入）

---

## 8. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 文言文可收集数据量低于预期 | 多源聚合；按朝代/类别做均衡采样避免分布偏移；接受数据上限，不为此牺牲数据质量 |
| 训练发散 | 先用极少量数据过拟合验证整条 pipeline；每 500 步 eval；梯度范数监控；学习率 warmup |
| 157M 模型对话质量有限 | 预期内——第 1 版重在跑通全流程；后续迭代方向：增大模型、混入现代中文数据、优化 SFT 数据质量 |
| Tokenizer 生僻字处理差 | byte_fallback 兜底；character_coverage=0.99995 |
| 数据源链接随时间失效 | 优先下载最大最稳定的语料；本地备份所有原始数据 |
