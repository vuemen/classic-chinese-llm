# 评估与评测模块设计文档

**所属阶段:** Phase 5 — 指令微调与评测
**涉及模块:** `src/classic_chinese_llm/evaluation/`
**日期:** 2026-07-28

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | Perplexity 评估 | 在 held-out 测试集上计算模型的困惑度 (PPL) |
| F2 | 生成质量指标 | BLEU-4、ROUGE-L、字符级准确率等 NLG 标准指标 |
| F3 | LLM-as-Judge 评测 | 使用评估准则/rubric 对模型生成结果进行质量打分 |
| F4 | 批量评测 | 支持从 JSONL 测试集批量加载样本，逐条生成并评测 |
| F5 | 评测报告 | 生成结构化的评测报告（控制台表格 + JSON 文件） |
| F6 | Task-specific 指标 | 针对文言文任务（翻译、诗词创作、问答）的专用质量检查 |

### 1.2 非功能需求

- **可扩展**: 新增评测指标无需修改 Evaluator 核心逻辑
- **确定性**: 使用固定 seed 和 temperature=0 时评测结果可复现
- **低内存**: 评测可在 CPU 上运行（加载 FP32 checkpoint），不强制需要 GPU
- **与训练解耦**: 评测模块独立于训练框架，接受 checkpoint 路径即可运行

---

## 2. 方案选型与对比

### 2.1 评测策略：自建 vs 复用

| 方案 | 灵活性 | 文言文适配 | 学习价值 | 维护成本 | 结论 |
|------|--------|-----------|---------|---------|------|
| **自建 EvaluationMetrics** | ⭐⭐⭐ | ✅ 可定制文言文专用指标 | ⭐⭐⭐ | ⭐⭐ | ✅ 选用 |
| sacrebleu / evaluate | ⭐⭐ | ⚠️ tokenizer 基于现代文本 | ⭐ | ⭐⭐⭐ | ❌ |
| lm-evaluation-harness | ⭐ | ❌ 需实现大量适配器 | ⭐ | ⭐ | ❌ |

**最终选择: 自建评测模块**。

### 2.2 LLM-as-Judge：外部 API vs 自建 Rubric

| 方案 | 成本 | 可用性 | 评测质量 | 结论 |
|------|------|--------|---------|------|
| **自建 Rubric + 关键词检查** | 零 | ✅ | ⭐⭐ | ✅ 选用 |
| 外部 LLM API (GPT-4) | 高 | ⚠️ 需网络+付费 | ⭐⭐⭐ | ❌ 违背项目原则 |
| 使用自身模型评判 | 零 | ✅ | ⭐ | ❌ 循环自评不可靠 |

**最终选择: 自建 Rubric 评测系统**。基于规则和关键词匹配的自动化评分，辅以人工可读的评测报告。

### 2.3 指标选择

| 指标 | 适用场景 | 是否引入 |
|------|---------|---------|
| **Perplexity** | 语言建模质量 | ✅ 核心指标 |
| **BLEU-4** | 翻译/生成质量（n-gram 匹配） | ✅ 引入 |
| **ROUGE-L** | 生成摘要/回答质量（LCS） | ✅ 引入 |
| **字符级准确率** | 逐字匹配度 | ✅ 引入 |
| **文言文专用检查** | 句式、虚词、典故 | ✅ 引入 |

---

## 3. 组件详细设计

### 3.1 Metrics (`evaluation/metrics.py`)

提供独立的指标计算函数，每个函数接受 `list[str]` (预测) 和 `list[list[str]]` (参考) 并返回 `float`。

```python
def calc_perplexity(loss: float) -> float:
    """将 cross-entropy loss 转换为 perplexity。"""
    return math.exp(loss)

def calc_bleu(predictions: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """计算 corpus-level BLEU-n 分数。"""

def calc_rouge_l(predictions: list[str], references: list[list[str]]) -> float:
    """计算 ROUGE-L (Longest Common Subsequence) F1 分数。"""

def calc_char_accuracy(predictions: list[str], references: list[list[str]]) -> float:
    """逐字符匹配准确率。"""

def calc_classical_chinese_score(prediction: str) -> dict[str, float]:
    """文言文质量评分（虚词密度、句式复杂度、典故覆盖率）。"""
```

### 3.2 Evaluator (`evaluation/evaluator.py`)

评测器：加载模型 → 加载测试集 → 逐条生成 → 计算指标 → 输出报告。

```python
class Evaluator:
    """模型评测器。

    职责:
    1. 从 checkpoint 加载模型 + tokenizer
    2. 加载测试数据集 (JSONL ChatML 或纯文本)
    3. 逐条生成回答
    4. 计算所有已注册的指标
    5. 输出评测报告
    """

    def __init__(self, model: nn.Module, tokenizer, generator, config: EvalConfig): ...
    def evaluate(self, test_data_path: Path) -> EvalReport: ...
    def _generate_responses(self, samples) -> list[EvalSample]: ...
    def _compute_metrics(self, samples) -> dict[str, float]: ...
```

### 3.3 EvalConfig (`evaluation/config.py`)

```python
@dataclass
class EvalConfig:
    max_samples: int = 500          # 评测样本上限
    metrics: list[str] = field(     # 启用的指标
        default_factory=lambda: ["perplexity", "bleu", "rouge_l", "char_accuracy"]
    )
    generation: GenerationConfig    # 生成参数（默认 temperature=0 确定性生成）
    output_dir: Path | None = None  # 报告输出目录
```

### 3.4 EvalReport (`evaluation/report.py`)

```python
@dataclass
class EvalSample:
    prompt: str
    reference: str
    prediction: str
    metrics: dict[str, float]

@dataclass
class EvalReport:
    config: EvalConfig
    samples: list[EvalSample]
    aggregate_metrics: dict[str, float]
    timestamp: str
    model_info: dict[str, Any]

    def to_json(self, path: Path) -> None: ...
    def summary(self) -> str: ...  # 人类可读的表格化总结
```

---

## 4. 模块结构

```
src/classic_chinese_llm/evaluation/
├── __init__.py       # 导出: Evaluator, EvalConfig, EvalReport, metrics
├── config.py         # EvalConfig
├── metrics.py        # 指标计算函数 (perplexity, BLEU, ROUGE-L, char_accuracy, classical_chinese_score)
├── evaluator.py      # Evaluator: 加载模型 → 批量生成 → 计算指标 → 输出报告
└── report.py         # EvalReport + EvalSample

tests/test_evaluation/
├── __init__.py
├── test_metrics.py   # 指标函数单元测试
├── test_evaluator.py # Evaluator 集成测试（使用小模型 + 虚拟数据）
└── test_report.py    # 报告生成测试
```

---

## 5. 接口定义汇总

```python
# metrics.py
def calc_perplexity(loss: float) -> float: ...
def calc_bleu(predictions: list[str], references: list[list[str]], max_n: int = 4) -> float: ...
def calc_rouge_l(predictions: list[str], references: list[list[str]]) -> float: ...
def calc_char_accuracy(predictions: list[str], references: list[list[str]]) -> float: ...
def calc_classical_chinese_score(prediction: str) -> dict[str, float]: ...

# evaluator.py
class Evaluator:
    def __init__(self, model, tokenizer, generator, config: EvalConfig) -> None: ...
    def evaluate(self, test_data_path: Path) -> EvalReport: ...

# config.py
@dataclass
class EvalConfig:
    max_samples: int
    metrics: list[str]
    generation: GenerationConfig
    output_dir: Path | None

# report.py
@dataclass
class EvalSample:
    prompt: str
    reference: str
    prediction: str
    metrics: dict[str, float]

@dataclass
class EvalReport:
    config: EvalConfig
    samples: list[EvalSample]
    aggregate_metrics: dict[str, float]
    timestamp: str
    model_info: dict[str, Any]
    def to_json(self, path: Path) -> None: ...
    def summary(self) -> str: ...
```

---

## 6. 验证清单

- [ ] 各指标函数对已知输入输出正确结果
- [ ] `calc_perplexity(1.0)` ≈ 2.718
- [ ] `calc_bleu` 对完全匹配的预测-参考返回接近 1.0
- [ ] `calc_rouge_l` 正确处理中文文本
- [ ] `calc_char_accuracy` 正确处理多参考
- [ ] `EvalReport.to_json()` 生成有效 JSON
- [ ] `EvalReport.summary()` 返回非空字符串
- [ ] Evaluator 在 CPU 上使用小模型完成一轮评测（≥2 个样本）
- [ ] 所有函数 ≤ 50 行
- [ ] 所有函数签名含完整类型注解
