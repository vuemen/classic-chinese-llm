"""评估与评测模块。

提供:
- EvalConfig: 评测配置
- EvalReport / EvalSample: 评测报告数据结构
- Evaluator: 模型评测器 (加载 → 生成 → 指标计算 → 报告)
- metrics: 指标计算函数 (perplexity, BLEU, ROUGE-L, char_accuracy)
"""

from classic_chinese_llm.evaluation.config import EvalConfig
from classic_chinese_llm.evaluation.evaluator import Evaluator
from classic_chinese_llm.evaluation.metrics import (
    calc_bleu,
    calc_char_accuracy,
    calc_classical_chinese_score,
    calc_perplexity,
    calc_rouge_l,
)
from classic_chinese_llm.evaluation.report import EvalReport, EvalSample

__all__ = [
    "EvalConfig",
    "EvalReport",
    "EvalSample",
    "Evaluator",
    "calc_bleu",
    "calc_char_accuracy",
    "calc_classical_chinese_score",
    "calc_perplexity",
    "calc_rouge_l",
]
