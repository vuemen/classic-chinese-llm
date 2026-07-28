"""对话界面层 —— Gradio Web UI + FastAPI REST API + 对话管理。

提供:
- Message / ConversationManager: 多轮对话管理
- SYSTEM_PROMPTS / get_system_prompt: 系统提示词模板
- create_app: FastAPI 应用工厂（OpenAI 兼容 API）
- create_ui: Gradio Blocks 界面创建
"""

from classic_chinese_llm.chat.api import create_app
from classic_chinese_llm.chat.app import create_ui
from classic_chinese_llm.chat.conversation import ConversationManager, Message
from classic_chinese_llm.chat.prompts import (
    SYSTEM_PROMPTS,
    get_system_prompt,
    list_system_prompts,
)

__all__ = [
    "ConversationManager",
    "Message",
    "SYSTEM_PROMPTS",
    "create_app",
    "create_ui",
    "get_system_prompt",
    "list_system_prompts",
]
