"""Tokenizer 模块。

提供文言文 SentencePiece Unigram Tokenizer 的完整工具链:
- TokenizerConfig: 训练配置
- TokenizerTrainer: 训练封装
- ClassicalChinesePreTokenizer: 文言文预分词器
- build_tokenizer: HF PreTrainedTokenizerFast 工厂函数
- save_tokenizer: Tokenizer 序列化
"""

from __future__ import annotations

from classic_chinese_llm.tokenizer.config import TokenizerConfig
from classic_chinese_llm.tokenizer.pretokenizer import (
    ClassicalChinesePreTokenizer,
    create_pretokenizer,
)
from classic_chinese_llm.tokenizer.trainer import TokenizerTrainer
from classic_chinese_llm.tokenizer.wrapper import (
    CHAT_TEMPLATE_JINJA,
    build_tokenizer,
    save_tokenizer,
)

__all__ = [
    "TokenizerConfig",
    "TokenizerTrainer",
    "ClassicalChinesePreTokenizer",
    "create_pretokenizer",
    "build_tokenizer",
    "save_tokenizer",
    "CHAT_TEMPLATE_JINJA",
]
