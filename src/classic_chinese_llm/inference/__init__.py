"""推理引擎 —— 模型加载 + 文本生成 + 流式输出。

提供:
- InferenceEngine: 高层推理封装，接受文本输入返回文本输出
"""

from classic_chinese_llm.inference.engine import InferenceEngine

__all__ = ["InferenceEngine"]
