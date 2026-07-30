# Classic Chinese LLM 场景使用指南

本文档按用户场景组织，从环境搭建到生产部署，覆盖项目的全部使用方式。请根据你的需求跳转到对应章节。

---

## 目录

- [场景零：我只想搭建环境](#场景零我只想搭建环境)
- [场景一：从零开始，完整走通全流程](#场景一从零开始完整走通全流程)
- [场景二：我只想训练（数据 → Tokenizer → 预训练 → SFT）](#场景二我只想训练数据--tokenizer--预训练--sft)
- [场景三：我只想推理（已有 checkpoint，直接用）](#场景三我只想推理已有-checkpoint直接用)
- [场景四：我想部署为 API 服务](#场景四我想部署为-api-服务)
- [场景五：我要开发调试（验证 Pipeline 是否跑通）](#场景五我要开发调试验证-pipeline-是否跑通)
- [场景六：我只需要处理数据](#场景六我只需要处理数据)
- [场景七：我只想训练或评估 Tokenizer](#场景七我只想训练或评估-tokenizer)
- [参考：配置项速查](#参考配置项速查)
- [参考：常见问题](#参考常见问题)
- [参考：深入阅读](#参考深入阅读)

---

## 场景零：我只想搭建环境

**适用人群**：第一次接触本项目，需要把开发运行环境搭起来。

### 1. 克隆项目

```bash
git clone https://github.com/vuemen/classic-chinese-llm.git
cd classic-chinese-llm
```

### 2. 创建 Conda 环境

```bash
conda create -n classic-llm python=3.12 -y
conda activate classic-llm
```

### 3. 安装 PyTorch

根据你的 CUDA 版本选择对应的安装命令。前往 [pytorch.org](https://pytorch.org/get-started/locally/) 确认。

```bash
# CUDA 12.4（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only（仅用于数据处理和推理测试，无法训练）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. 安装项目依赖

```bash
# 一次性安装全部依赖（推荐）
pip install -r requirements.txt
```

`requirements.txt` 中已按功能分组并标注版本号。如果你不需要某组依赖（比如不做数据采集），可以编辑该文件注释掉对应部分。

### 5. 注册项目模块

本项目使用 `src` 布局，`classic_chinese_llm` 包位于 `src/` 下，需要让 Python 能找到它。**任选以下一种方式**：

**方式一：可编辑安装（推荐，一次配置永久生效）**

```bash
pip install -e .
```

之后可以直接运行脚本，无需额外配置。

**方式二：设置 `PYTHONPATH`（免安装，每次打开终端需重新设置）**

```bash
# Linux / macOS / Git Bash（在项目根目录下执行）
export PYTHONPATH="$(pwd)/src"

# Windows PowerShell（在项目根目录下执行）
$env:PYTHONPATH = "$(Get-Location)\src"
```

也可以在运行命令时临时指定：

```bash
PYTHONPATH="$(pwd)/src" python scripts/collect_data.py --raw-dir data/raw       # Linux / Git Bash
$env:PYTHONPATH="$(Get-Location)\src"; python scripts/collect_data.py --raw-dir data/raw  # PowerShell
```

### 6. 验证

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import classic_chinese_llm; print('✅ 核心模块加载成功')"
```

如果上述命令均无报错，环境即搭建完毕。

---

## 场景一：从零开始，完整走通全流程

**适用人群**：想从头体验整个 LLM 训练流程——数据处理、分词器训练、预训练、指令微调、对话。

**预计总耗时**：2-4 天（含预训练 2-3 天）

### 流程概览

```
[数据采集] → [数据清洗] → [数据去重] → [Tokenizer训练]
                                              │
                                              ▼
                              [预训练] ← [预训练数据集]
                                 │
                                 ▼
[指令数据集] → [指令微调 (SFT)] → [对话 / API 服务]
```

#### 各阶段中间产物与跨机器可移植性

| 阶段 | 中间产物 | 格式 | 可跨机器 | 说明 |
|------|---------|------|:---:|------|
| 数据采集 | `data/raw/*.jsonl` | JSONL 文本 | ✅ | 纯文本，任意 OS 通用 |
| 数据清洗 | `data/processed/cleaned.txt` | 纯文本 (UTF-8) | ✅ | 纯文本，任意 OS 通用 |
| 数据去重 | `data/processed/deduplicated.jsonl` | JSONL 文本 | ✅ | 纯文本，任意 OS 通用 |
| 指令数据集 | `data/processed/instructions/*.jsonl` | JSONL 文本 | ✅ | 纯文本，任意 OS 通用 |
| Tokenizer 训练 | `models/tokenizer/classical_chinese.model` | SentencePiece 二进制 | ✅ | protobuf 格式，跨平台通用 |
| 预训练 | `models/checkpoints/checkpoint_*.pt` | PyTorch state_dict | ⚠️ | 见下方说明 |
| 指令微调 (SFT) | `models/checkpoints/sft_best.pt` | PyTorch state_dict | ⚠️ | 见下方说明 |

> **⚠️ Checkpoint 跨机器使用须知：**
> - **CPU ↔ GPU**：在 GPU 上保存的 checkpoint 可以通过 `map_location='cpu'` 在 CPU 机器上加载，反之亦可
> - **架构匹配**：checkpoint 与模型配置（`d_model`、`n_layers`、`n_heads` 等）强绑定，加载时必须使用相同的模型配置
> - **PyTorch 版本**：`torch.save`/`torch.load` 向前兼容性良好（2.x 可加载 1.x），跨小版本通常无问题
> - **操作系统**：`.pt` 文件跨 Windows / Linux / macOS 通用
>
> **典型跨机器协作流程**：在 CPU 机器上完成数据采集→清洗→去重→Tokenizer 训练，将 `data/processed/` 和 `models/tokenizer/` 拷贝到 GPU 机器上做预训练和 SFT，最后将 checkpoint 拷回任意机器做推理部署。

### 前置准备：下载原始语料

**`collect_data.py` 不会自动下载数据**，它只扫描本地已有文件进行解析。你需要先将原始文件放到 `data/raw/<source>/` 下。

#### 数据源下载指南

**殆知阁**（主力语料，~20 亿字，必下）：

```bash
# 方式 A：Git 克隆（推荐）
git clone --branch data --depth 1 https://github.com/frankslin/daizhigev20.git data/raw/daizhige

# 方式 B：GitHub ZIP
# 打开 https://github.com/garychowcmu/daizhigev20 → Code → Download ZIP → 解压到 data/raw/daizhige/

# 备选镜像（内容相同）：
# https://github.com/Will-learning-nlp/daizhigev20
# https://github.com/mrsunx/daizhigev20
```

> 殆知阁收录约 16,000 种古籍、20 万卷，涵盖经/史/子/集/诗/艺/易/医/佛/道十大类，格式为 `.txt`。

**WikiSource 中文**（维基文库 XML dump）：

```bash
# 下载最新 pages-articles dump
# 地址：https://dumps.wikimedia.org/zhwikisource/latest/
# 找到 zhwikisource-<日期>-pages-articles.xml.bz2 下载
# 放到 data/raw/wikisource/ 下（无需解压，适配器自动处理）
```

> `.xml.bz2` 文件通常几百 MB，`lxml.iterparse` 流式解析，内存友好。

**GitHub 开源语料**（可选扩展）：

| 项目 | 格式 | 说明 |
|------|------|------|
| [NiuTrans/Classical-Modern](https://github.com/NiuTrans/Classical-Modern) | `.txt` | 327 本书、97 万文言-白话平行句对 |
| [HistoryTrans/Dataset](https://huggingface.co/datasets/HistoryTrans/Dataset) | `.jsonl` | 二十四史+清史稿翻译数据 |

```bash
git clone --depth 1 https://github.com/NiuTrans/Classical-Modern.git data/raw/github/NiuTrans
```

**四库全书**（可选，公共领域子集）：

```bash
# Project Gutenberg 上的文渊阁四库全书（免费、公共领域）
git clone --depth 1 https://github.com/GITenberg/-------_7221.git data/raw/siku/gutenberg
```

> ⚠️ Project Gutenberg 版仅覆盖部分内容。更完整的四库全书电子版获取较困难，大部分为付费资源或高校馆内使用。适配器按"经/史/子/集"目录结构推断体裁——如果你有更多内容，按目录归类放置即可。

**ctext.org**（可选，高质量补充）：

1. 注册 [ctext.org](https://ctext.org/zh) 免费账户
2. 登录后安装"全文输出"插件
3. 逐章手动下载需要的典籍 TXT，放到 `data/raw/ctext/`

> ⚠️ ctext.org **禁止自动批量抓取**，违者会被封锁。精选几本最重要的经典（如《论语》《孟子》《庄子》《史记》）即可。

#### 优先级建议

```
殆知阁 > WikiSource > GitHub语料 > 四库全书 > ctext
  必下      推荐        可选        可选     精选几本
```

只需下载殆知阁即可跑通全流程，其他数据源逐步补充。

#### 目录结构要求

```
data/raw/
├── daizhige/       # .txt 文件（可含子目录）
├── wikisource/     # .xml 或 .xml.bz2
├── github/         # .txt 或 .jsonl（按项目建子目录）
├── siku/           # .txt 文件（建议按 经/史/子/集 建子目录）
└── ctext/          # .txt 文件
```

### 第一步：采集并处理文言文语料

```bash
python scripts/collect_data.py --raw-dir data/raw
```

这一步会扫描 `data/raw/` 下的原始文件，经过采集（解析→校验）、清洗（Unicode 规范化、现代标点剥离、版式噪声去除、长度过滤）、去重（SHA-256 精确去重 + MinHash LSH 近似去重）三个阶段，最终在 `data/processed/` 下生成处理后的数据：

```
data/processed/
├── cleaned.txt          # 清洗后的纯文本（用于 Tokenizer 训练）
├── deduplicated.jsonl   # 去重后的 JSONL（用于预训练）
└── instructions/        # 指令数据集（用于 SFT）
    ├── train.jsonl
    └── val.jsonl
```

### 第二步：训练分词器

```bash
python scripts/train_tokenizer.py --corpus data/processed/cleaned.txt --vocab-size 32000
```

训练完成后，分词器模型保存在 `models/tokenizer/classical_chinese.model`。HuggingFace 格式的封装也同时导出，可与 `datasets` 库无缝互操作。

可通过以下方式验证分词效果：

```bash
python -c "
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer
tokenizer = build_tokenizer('models/tokenizer/classical_chinese.model')
print(tokenizer.encode('子曰学而时习之不亦说乎'))
print(tokenizer.decode(tokenizer.encode('子曰学而时习之不亦说乎')))
"
```

### 第三步：预训练

```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

**预计耗时**：2-3 天（12GB GPU，batch_size=8，gradient_accumulation=4）

训练过程会每 500 步输出一次 eval loss，每 2000 步保存一个 checkpoint（保留最近 5 个）。日志输出到 `logs/pretrain.log`。

**常用变体**：

```bash
# 从 checkpoint 恢复训练（中断后继续）
python scripts/pretrain.py --config configs/pretrain.yaml --resume

# 调整 batch size（通过环境变量覆盖 YAML 配置）
CCLLM_TRAINING__BATCH_SIZE=4 CCLLM_TRAINING__GRADIENT_ACCUMULATION_STEPS=8 \
    python scripts/pretrain.py --config configs/pretrain.yaml

# 指定训练数据路径
python scripts/pretrain.py --config configs/pretrain.yaml --data-path data/custom.jsonl
```

训练完成后，在 `models/checkpoints/` 下可以找到 checkpoint 文件。选择 eval loss 最低的 checkpoint 用于后续步骤。

### 第四步：指令微调

```bash
python scripts/finetune.py \
    --config configs/sft.yaml \
    --pretrained-checkpoint models/checkpoints/checkpoint_best.pt
```

**预计耗时**：2-4 小时（12GB GPU）

SFT 阶段仅计算 assistant 回复部分的 loss（通过 label masking 实现），避免模型在用户输入上浪费学习能力。训练日志输出到 `logs/sft.log`。

```bash
# 使用自定义训练/验证数据
python scripts/finetune.py \
    --config configs/sft.yaml \
    --pretrained-checkpoint models/checkpoints/checkpoint_best.pt \
    --train-data data/my_instructions/train.jsonl \
    --val-data data/my_instructions/val.jsonl
```

### 第五步：对话

微调完成后，即可启动对话界面：

```bash
# Gradio Web UI（浏览器中打开 http://localhost:7860）
python scripts/chat.py --checkpoint models/checkpoints/sft_best.pt

# 命令行对话
python scripts/chat.py --checkpoint models/checkpoints/sft_best.pt --mode cli
```

---

## 场景二：我只想训练（数据 → Tokenizer → 预训练 → SFT）

**适用人群**：已有训练数据，只需要跑训练流程；或者只想重新训练模型。

**预计耗时**：2-4 天

### 如果你已经有清洗好的数据

跳过数据采集，直接进入后续步骤：

```bash
# 假设你的清洗后文本在 data/processed/cleaned.txt
# 假设你的指令数据集在 data/processed/instructions/

# 1. Tokenizer 训练
python scripts/train_tokenizer.py --corpus data/processed/cleaned.txt --vocab-size 32000

# 2. 预训练
python scripts/pretrain.py --config configs/pretrain.yaml

# 3. 指令微调（预训练完成后）
python scripts/finetune.py \
    --config configs/sft.yaml \
    --pretrained-checkpoint models/checkpoints/checkpoint_best.pt
```

### 如果你只想做预训练（不做 SFT）

```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

预训练完成后，可以直接用 checkpoint 做文本续写（不经过对话格式）。参见 [场景三](#场景三我只想推理已有-checkpoint直接用) 的 Python API 调用示例。

### 如果你只想做 SFT（已有预训练 checkpoint）

```bash
python scripts/finetune.py \
    --config configs/sft.yaml \
    --pretrained-checkpoint models/checkpoints/pretrain_best.pt
```

### 训练监控

训练过程中的关键指标：

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| train/loss | 训练损失 | 持续下降 |
| eval/loss | 验证损失 | 持续下降，不高于 train/loss 太多 |
| eval/perplexity | 困惑度 | 持续下降 |
| lr | 当前学习率 | 随 warmup 上升，随 cosine decay 下降 |
| grad_norm | 梯度范数 | < 10，突然飙升可能发散 |

如果 eval/loss 不再下降但 train/loss 在降，说明开始过拟合，可以提前停止。

---

## 场景三：我只想推理（已有 checkpoint，直接用）

**适用人群**：已经训练好了模型，或者从别处拿到了 checkpoint 和 tokenizer，只想要对话。

**要求**：具备以下文件：
- 模型 checkpoint（`.pt` 文件）
- Tokenizer 模型（`models/tokenizer/classical_chinese.model`）

### Gradio Web 对话

```bash
python scripts/chat.py \
    --checkpoint models/checkpoints/sft_best.pt \
    --tokenizer models/tokenizer/classical_chinese.model
```

浏览器打开 `http://localhost:7860` 即可开始对话。界面上可调节 **Temperature**、**Top-P**、**最大生成长度** 等参数。

### 命令行对话

```bash
python scripts/chat.py \
    --checkpoint models/checkpoints/sft_best.pt \
    --mode cli
```

### Python API 调用

```python
from classic_chinese_llm.inference.engine import InferenceEngine
from classic_chinese_llm.model.transformer import TransformerLM
from classic_chinese_llm.model.generation import GenerationConfig
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer
from classic_chinese_llm.utils.checkpoint import load_checkpoint
from classic_chinese_llm.utils.device import detect_device
import torch

# 1. 加载 tokenizer
tokenizer = build_tokenizer("models/tokenizer/classical_chinese.model")

# 2. 加载模型
device = detect_device()
model = TransformerLM(...)  # 或从 config 创建
load_checkpoint("models/checkpoints/sft_best.pt", model, device=device)
model.eval()

# 3. 创建推理引擎
engine = InferenceEngine(
    model=model,
    tokenizer_encode_fn=tokenizer.encode,
    tokenizer_decode_fn=tokenizer.decode,
    generation_config=GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=512),
)

# 4. 生成
response = engine.generate("子曰学而时习之")
print(response)
```

### 可调生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 0.7 | 温度越高输出越多样，越低越确定。（0 为贪婪解码） |
| `top_p` | 0.9 | Nucleus sampling，只从累积概率超过该值的 token 中采样 |
| `top_k` | 50 | 只从概率最高的 K 个 token 中采样 |
| `max_new_tokens` | 512 | 最大生成长度 |
| `repetition_penalty` | 1.1 | >1 惩罚重复 token，<1 鼓励重复 |

---

## 场景四：我想部署为 API 服务

**适用人群**：想把模型部署为 HTTP API 服务，供其他应用调用。支持 OpenAI 兼容格式。

### 启动 API 服务

```bash
python scripts/serve.py \
    --checkpoint models/checkpoints/sft_best.pt \
    --host 0.0.0.0 \
    --port 8000
```

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容的对话接口 |
| `/v1/chat/completions` | POST (stream) | SSE 流式响应（设置 `stream: true`） |
| `/health` | GET | 健康检查 |

### 调用示例

```bash
# 非流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "classical-chinese-llm",
    "messages": [
      {"role": "user", "content": "以文言文写一首描写春天的诗"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# 流式 (SSE)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "classical-chinese-llm",
    "messages": [
      {"role": "user", "content": "何为君子？"}
    ],
    "stream": true
  }'
```

### 使用 OpenAI SDK 调用

因为 API 是 OpenAI 兼容的，可以直接用 OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="classical-chinese-llm",
    messages=[
        {"role": "system", "content": "你是一位精通中国古代文学的助手，请用文言文回答。"},
        {"role": "user", "content": "请给我讲讲孔子的仁学思想。"},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

---

## 场景五：我要开发调试（验证 Pipeline 是否跑通）

**适用人群**：开发者，想在正式训练前快速验证 pipeline 是否正常工作。

### 快速验证流程（小规模）

```bash
# 1. 用少量数据快速训练 tokenizer
python scripts/train_tokenizer.py \
    --corpus data/processed/cleaned.txt \
    --vocab-size 4000 \
    --output models/tokenizer/debug

# 2. 小模型 + 少量 step 验证预训练 pipeline
CCLLM_MODEL__N_LAYERS=2 CCLLM_MODEL__D_MODEL=256 CCLLM_MODEL__N_HEADS=4 \
CCLLM_TRAINING__MAX_STEPS=100 CCLLM_TRAINING__EVAL_EVERY=20 \
    python scripts/pretrain.py --config configs/pretrain.yaml

# 3. 验证 SFT pipeline（需要先有预训练 checkpoint）
CCLLM_TRAINING__MAX_EPOCHS=1 CCLLM_TRAINING__EVAL_EVERY=10 \
    python scripts/finetune.py \
    --config configs/sft.yaml \
    --pretrained-checkpoint models/checkpoints/checkpoint_best.pt
```

### 运行测试

```bash
# 全部测试
pytest tests/ -v

# 单个模块测试
pytest tests/test_model/ -v        # 模型层测试
pytest tests/test_data/ -v         # 数据管道测试
pytest tests/test_tokenizer/ -v    # Tokenizer 测试
pytest tests/test_training/ -v     # 训练测试

# 带覆盖率报告
pytest tests/ -v --cov=src/classic_chinese_llm --cov-report=html
```

### 代码质量检查

```bash
black src/ tests/          # 格式化
ruff check src/ tests/     # Lint
mypy src/                  # 类型检查
```

### 调试技巧

```bash
# 提升日志级别为 DEBUG，查看详细训练信息
CCLLM_LOGGING__LEVEL=DEBUG python scripts/pretrain.py --config configs/pretrain.yaml

# 过拟合测试：用极小数据验证模型和学习率是否正常
# 创建一个只有 100 行的 mini.jsonl，训练 50 step，loss 应该迅速降到接近 0
CCLLM_TRAINING__MAX_STEPS=50 \
    python scripts/pretrain.py --config configs/pretrain.yaml --data-path data/mini.jsonl
```

---

## 场景六：我只需要处理数据

**适用人群**：只需要采集、清洗文言文数据，不打算训练模型；或者想为自己的项目准备训练数据。

> ⚠️ **前置条件**：原始语料需手动下载到 `data/raw/<source>/` 下，详见[前置准备：下载原始语料](#前置准备下载原始语料)。脚本不会自动下载数据。

### 完整流程一键运行

```bash
python scripts/collect_data.py --raw-dir data/raw
```

### 分步执行（高级）

如果需要单独控制采集、清洗、去重、格式化的每个步骤，可以通过 Python API：

```python
from classic_chinese_llm.data.collector import DataCollector
from classic_chinese_llm.data.cleaner import DataCleaner
from classic_chinese_llm.data.deduplicator import DataDeduplicator
from classic_chinese_llm.data.formatter import InstructionFormatter

# 1. 采集
collector = DataCollector(output_dir="data/raw")
collector.collect_all()  # 或 collector.collect("daizhige") 指定单个源

# 2. 清洗
cleaner = DataCleaner()
cleaned = cleaner.clean("data/raw/daizhige.jsonl")
cleaner.save(cleaned, "data/processed/cleaned_daizhige.jsonl")

# 3. 去重
dedup = DataDeduplicator()
dedup.deduplicate("data/processed/cleaned_merged.jsonl", "data/processed/deduplicated.jsonl")

# 4. 构建指令数据集
formatter = InstructionFormatter()
formatter.format("data/processed/deduplicated.jsonl", "data/processed/instructions/")
```

### 数据源配置

在配置中启用或禁用特定数据源：

```python
from classic_chinese_llm.config.settings import CollectorConfig

config = CollectorConfig(
    enabled_sources=["daizhige", "wikisource"],  # 仅启用这两个源
    retry_attempts=5,                             # 重试次数
)
```

---

## 场景七：我只想训练或评估 Tokenizer

**适用人群**：只需要训练、评估或使用文言文分词器。

### 训练分词器

```bash
# 默认配置（32K vocab，Unigram 模型）
python scripts/train_tokenizer.py --corpus data/processed/cleaned.txt --vocab-size 32000

# 自定义参数
python scripts/train_tokenizer.py \
    --corpus data/processed/cleaned.txt \
    --vocab-size 16000 \
    --character-coverage 0.9995 \
    --output models/tokenizer/custom
```

### 评估分词器

```python
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer

tokenizer = build_tokenizer("models/tokenizer/classical_chinese.model")

# 查看词表大小
print(f"Vocab size: {tokenizer.vocab_size}")

# 编码 / 解码
text = "子曰学而时习之不亦说乎"
ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(ids)}")
print(f"Decoded: {tokenizer.decode(ids)}")

# 测试特殊字符处理
test_texts = [
    "道可道非常道",        # 基础文言
    "《詩經·國風·周南》",   # 书名号
    "𣛧𤩲𦬊",               # 生僻字（UTF-8 扩展区）
]
for t in test_texts:
    ids = tokenizer.encode(t)
    decoded = tokenizer.decode(ids)
    print(f"  '{t}' → {len(ids)} tokens, roundtrip OK: {t == decoded}")
```

---

## 参考：配置项速查

### 环境变量覆盖

所有 YAML 配置项都可以通过环境变量覆盖，前缀 `CCLLM_`，嵌套用 `__` 分隔：

```bash
# 修改模型架构
CCLLM_MODEL__N_LAYERS=8 CCLLM_MODEL__D_MODEL=512 python scripts/pretrain.py ...

# 修改训练参数
CCLLM_TRAINING__BATCH_SIZE=16 CCLLM_TRAINING__LEARNING_RATE=1e-4 python scripts/pretrain.py ...

# 修改日志级别
CCLLM_LOGGING__LEVEL=DEBUG python scripts/pretrain.py ...
```

### 预训练关键配置 (`configs/pretrain.yaml`)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `training.batch_size` | 8 | 每 GPU batch size |
| `training.gradient_accumulation_steps` | 4 | 梯度累积步数（有效 batch = 8×4 = 32） |
| `training.learning_rate` | 3e-4 | 初始学习率 |
| `training.max_steps` | 100000 | 总训练步数 |
| `training.warmup_steps` | 1000 | 学习率预热步数 |
| `training.eval_every` | 500 | 每隔多少步验证一次 |
| `training.save_every` | 2000 | 每隔多少步保存 checkpoint |
| `optimizer.betas` | [0.9, 0.95] | Adam β 参数 |
| `scheduler.min_lr` | 3e-5 | 学习率衰减最小值 |

### SFT 关键配置 (`configs/sft.yaml`)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `training.batch_size` | 4 | 每 GPU batch size |
| `training.gradient_accumulation_steps` | 8 | 有效 batch = 4×8 = 32 |
| `training.learning_rate` | 1e-4 | 初始学习率（比预训练低） |
| `training.max_epochs` | 3 | 训练轮数 |
| `training.warmup_steps` | 100 | 预热步数（SFT 预热较短） |
| `data.max_samples` | 15000 | 最大样本数 |
| `data.val_split` | 0.05 | 验证集比例 |

---

## 参考：常见问题

### 显存不足 (CUDA Out of Memory)

```bash
# 减小 batch size，增大梯度累积
CCLLM_TRAINING__BATCH_SIZE=2 CCLLM_TRAINING__GRADIENT_ACCUMULATION_STEPS=16 \
    python scripts/pretrain.py --config configs/pretrain.yaml
```

### 训练过程中 loss 变成 NaN

可能原因：学习率过高、梯度爆炸。尝试：

```bash
# 降低学习率
CCLLM_TRAINING__LEARNING_RATE=1e-4 python scripts/pretrain.py --config configs/pretrain.yaml
```

然后检查 `logs/pretrain.log` 中的 `grad_norm` 是否异常。

### Tokenizer 加载失败

确保 tokenizer 文件存在于 `models/tokenizer/` 下：

```
models/tokenizer/
├── classical_chinese.model      # SentencePiece 模型
├── tokenizer.json                # HF tokenizer 封装
└── ...
```

### 数据采集失败

某些数据源可能因网络原因无法访问。可以单独禁用某个源：

```python
# 在 src/classic_chinese_llm/config/settings.py 的 CollectorConfig 中
# 修改 enabled_sources 的默认值，或在代码中指定
```

### Windows 路径问题

使用 PowerShell 时，路径分隔符兼容 `/` 和 `\`。建议统一使用 `/`：

```powershell
python scripts/pretrain.py --config configs/pretrain.yaml --data-path data/my_corpus.jsonl
```

---

## 参考：深入阅读

| 文档 | 说明 |
|------|------|
| [架构设计文档](architecture.md) | 项目整体架构、分层定义、技术栈、模型规模与数据约束 |
| [模型层设计](design/09-model.md) | 完整架构图、方案选型对比（RoPE/SwiGLU/Pre-norm）、参数量逐项计算、显存估算 |
| [Tokenizer 设计](design/08-tokenizer.md) | Unigram 算法选择、预分词规则、HF 封装、byte_fallback 机制 |
| [训练层设计](design/10-training.md) | Trainer 框架、预训练/SFT 流程、回调系统、显存预算分析 |
| [推理引擎设计](design/12-inference.md) | 模型加载、流式生成、KV Cache |
| [对话界面设计](design/13-chat.md) | Gradio UI、FastAPI、对话管理、OpenAI 兼容 API |
| [评估模块设计](design/11-evaluation.md) | Perplexity、NLG 指标、LLM-as-Judge、评测报告 |
| [全部设计文档](design/) | 共 13 份模块详细设计文档 |
