# 对话界面层设计文档

**所属阶段:** Phase 6 — 对话界面
**涉及模块:** `src/classic_chinese_llm/chat/`
**日期:** 2026-07-28

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | Gradio Web UI | 文言文风格对话界面，支持参数调节（temperature, top-p, top-k 等） |
| F2 | FastAPI REST API | `POST /v1/chat/completions` OpenAI 兼容接口，支持 SSE 流式响应 |
| F3 | 对话管理 | 多轮对话历史维护，超长上下文自动截断 |
| F4 | 系统提示词 | 预设角色模板：古文专家、诗词创作、历史讲述、文言翻译 |
| F5 | 参数透传 | UI 和 API 层将 generation 参数透传给 InferenceEngine |

### 1.2 非功能需求

- Gradio UI 可本地运行（无需外部服务）
- FastAPI 支持 CORS（允许前端跨域访问）
- 所有对话状态仅在内存中（不持久化），刷新即丢失
- API 响应格式兼容 OpenAI Chat Completions API

---

## 2. 方案选型与对比

### 2.1 Chat UI: Gradio vs Streamlit vs 自定义 HTML

| 方案 | 学习价值 | 开发效率 | 文言文 UI 定制 | 结论 |
|------|---------|---------|---------------|------|
| **Gradio Blocks** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ 可自定义 CSS | ✅ 选用 |
| Streamlit | ⭐⭐ | ⭐⭐⭐ | ⭐ | ❌ |
| 自定义 HTML/JS | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ❌ 时间成本 |

### 2.2 API: FastAPI SSE vs WebSocket

| 方案 | OpenAI 兼容 | 实现复杂度 | 客户端兼容性 | 结论 |
|------|-----------|-----------|-------------|------|
| **FastAPI + SSE** | ✅ 原生匹配 | ⭐⭐ | ✅ 广泛支持 | ✅ 选用 |
| WebSocket | ❌ 需自定义协议 | ⭐⭐⭐ | ⭐⭐ | ❌ |

---

## 3. 组件详细设计

### 3.1 对话管理 (`chat/conversation.py`)

```python
@dataclass
class Message:
    role: str       # "system" | "user" | "assistant"
    content: str

class ConversationManager:
    """多轮对话管理。

    职责:
    1. 维护消息历史 (system prompt + user/assistant 交替)
    2. 自动截断超长上下文（保留 system + 最近的 N 轮）
    3. 转换为 InferenceEngine 所需的 messages 格式
    """

    def __init__(self, system_prompt: str = "", max_turns: int = 20): ...
    def add_user(self, content: str) -> None: ...
    def add_assistant(self, content: str) -> None: ...
    def get_messages(self) -> list[dict[str, str]]: ...
    def reset(self) -> None: ...
    def set_system_prompt(self, prompt: str) -> None: ...
```

### 3.2 系统提示词 (`chat/prompts.py`)

```python
SYSTEM_PROMPTS: dict[str, str] = {
    "古文专家": "你是一位精通中国古代典籍的学者...",
    "诗词创作": "你是一位擅长创作古典诗词的诗人...",
    "历史讲述": "你是一位熟悉中国历史的说书人...",
    "文言翻译": "你是一位擅长文言文与现代汉语互译的翻译家...",
}

def get_system_prompt(name: str) -> str: ...
def list_system_prompts() -> list[str]: ...
```

### 3.3 FastAPI (`chat/api.py`)

```python
# 请求/响应模型 (OpenAI 兼容)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "classical-chinese-llm"
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 256
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]

# SSE 端点
@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest): ...
```

### 3.4 Gradio UI (`chat/app.py`)

```python
def create_ui(engine: InferenceEngine) -> gr.Blocks:
    """创建 Gradio Blocks 界面。

    布局:
    - 左侧: 模型参数面板 (temperature, top-p, top-k, max_tokens)
    - 右侧: 对话窗口 + 输入框
    - 顶部: 系统提示词选择下拉框
    """
```

---

## 4. 模块结构

```
src/classic_chinese_llm/chat/
├── __init__.py        # 导出主要接口
├── conversation.py    # Message, ConversationManager
├── prompts.py         # SYSTEM_PROMPTS, get_system_prompt
├── api.py             # FastAPI app + /v1/chat/completions 端点
└── app.py             # Gradio Blocks UI

tests/test_chat/
├── __init__.py
├── test_conversation.py
├── test_prompts.py
└── test_api.py
```

---

## 5. OpenAI 兼容 API 设计

请求格式:
```json
{
    "model": "classical-chinese-llm",
    "messages": [
        {"role": "system", "content": "你是一位文言文专家"},
        {"role": "user", "content": "请解释'学而时习之'的含义"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
}
```

非流式响应:
```json
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1719000000,
    "model": "classical-chinese-llm",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}]
}
```

流式响应 (SSE):
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"学"},"finish_reason":null}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"而"},"finish_reason":null}]}
...
data: [DONE]
```

---

## 6. 验证清单

- [ ] ConversationManager 正确维护消息历史
- [ ] 超长上下文自动截断（保留 system + 最近 N 轮）
- [ ] get_system_prompt 返回有效 prompt 文本
- [ ] list_system_prompts 返回非空列表
- [ ] FastAPI /v1/chat/completions 非流式返回 200
- [ ] FastAPI /v1/chat/completions 流式返回 SSE 事件
- [ ] Gradio UI 创建成功（Blocks 实例）
- [ ] 无效 prompt 名称抛出 KeyError
- [ ] API 请求体验证：缺少 messages 字段返回 422
