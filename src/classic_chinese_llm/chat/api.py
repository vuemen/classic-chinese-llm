"""FastAPI REST API —— OpenAI 兼容的聊天接口。

提供:
- POST /v1/chat/completions: 聊天补全（支持 SSE 流式）
- GET /v1/models: 可用模型列表
- GET /health: 健康检查
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from classic_chinese_llm.inference.engine import InferenceEngine
from classic_chinese_llm.model.generation import GenerationConfig
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic 模型
# ═══════════════════════════════════════════════════════════════════════════


class ChatMessage(BaseModel):
    """单条聊天消息。"""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """聊天补全请求（OpenAI 兼容格式）。"""

    model: str = "classical-chinese-llm"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    stream: bool = False

    @model_validator(mode="after")
    def _check_messages_not_empty(self) -> ChatCompletionRequest:
        """验证 messages 非空。"""
        if not self.messages:
            raise ValueError("messages 不能为空")
        return self


class ChatCompletionResponse(BaseModel):
    """聊天补全响应。"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI 应用
# ═══════════════════════════════════════════════════════════════════════════


def create_app(engine: InferenceEngine) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        engine: 推理引擎实例。

    Returns:
        FastAPI: 配置完成的 FastAPI 应用。
    """
    app = FastAPI(
        title="Classical Chinese LLM API",
        description="文言文大语言模型 OpenAI 兼容 API",
        version="0.1.0",
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ═══════════════════════════════════════════════════════════════════
    # 端点
    # ═══════════════════════════════════════════════════════════════════

    @app.get("/health")
    async def health() -> dict[str, str]:
        """健康检查端点。"""
        return {"status": "ok", "model": "classical-chinese-llm"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, list[dict[str, str]]]:
        """列出可用模型。"""
        return {
            "data": [
                {
                    "id": "classical-chinese-llm",
                    "object": "model",
                    "owned_by": "classic-chinese-llm",
                }
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        """聊天补全端点（支持 SSE 流式）。

        Args:
            request: ChatCompletionRequest 对象。

        Returns:
            非流式: ChatCompletionResponse dict。
            流式: StreamingResponse (SSE 格式)。
        """
        logger.info(
            "收到聊天请求: model=%s, messages=%d, stream=%s",
            request.model,
            len(request.messages),
            request.stream,
        )

        # 提取 messages 中的内容
        chat_history: list[dict[str, str]] = []
        user_content = ""

        for chat_msg in request.messages:
            d = {"role": chat_msg.role, "content": chat_msg.content}
            if chat_msg.role == "user":
                user_content = chat_msg.content
            chat_history.append(d)

        if not user_content:
            raise HTTPException(status_code=400, detail="至少需要一条 user 消息")

        # 将除最后一条 user 消息外的内容视为 history
        prompt_messages: list[dict[str, str]] = []
        for history_msg in chat_history:
            if history_msg["role"] == "user" and history_msg["content"] == user_content:
                break
            prompt_messages.append(history_msg)

        gen_config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
        )

        if request.stream:
            return _stream_response(
                engine=engine,
                prompt=user_content,
                history=prompt_messages,
                gen_config=gen_config,
                model_name=request.model,
            )

        # 非流式
        generated = engine.generate(
            prompt=user_content,
            history=prompt_messages,
            generation_config=gen_config,
        )

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated},
                    "finish_reason": "stop",
                }
            ],
        }
        return response

    return app


# ═══════════════════════════════════════════════════════════════════════════
# SSE 流式响应
# ═══════════════════════════════════════════════════════════════════════════


def _stream_response(
    engine: InferenceEngine,
    prompt: str,
    history: list[dict[str, str]],
    gen_config: GenerationConfig,
    model_name: str,
) -> StreamingResponse:
    """构建 SSE 流式响应。

    Args:
        engine: 推理引擎。
        prompt: 用户输入。
        history: 对话历史。
        gen_config: 生成参数。
        model_name: 模型名称。

    Returns:
        StreamingResponse: SSE 流式响应。
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def event_stream() -> Any:
        """SSE 事件生成器。"""
        for token_text in engine.stream(
            prompt=prompt,
            history=history,
            generation_config=gen_config,
        ):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {_json_dumps(chunk)}\n\n"

        # 发送 [DONE] 信号
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _json_dumps(obj: Any) -> str:
    """不依赖 json 模块的快速 JSON 序列化（用于流式 chunk）。"""
    import json

    return json.dumps(obj, ensure_ascii=False)
