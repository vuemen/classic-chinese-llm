"""chat.api 模块的单元测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from classic_chinese_llm.chat.api import (
    ChatCompletionRequest,
    ChatMessage,
    create_app,
)
from classic_chinese_llm.config.settings import ModelConfig
from classic_chinese_llm.inference.engine import InferenceEngine
from classic_chinese_llm.model.transformer import TransformerLM


def _make_test_engine() -> InferenceEngine:
    """创建测试用的推理引擎。"""
    model = TransformerLM(
        ModelConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            max_seq_len=128,
        )
    )
    return InferenceEngine(
        model=model,
        tokenizer_decode_fn=lambda ids: "文言文回答内容",
        tokenizer_encode_fn=lambda text: [1, 2, 3],
    )


class TestChatModels:
    """Pydantic 模型测试。"""

    def test_chat_message_creation(self) -> None:
        """ChatMessage 基本创建。"""
        msg = ChatMessage(role="user", content="你好")
        assert msg.role == "user"
        assert msg.content == "你好"

    def test_chat_completion_request_defaults(self) -> None:
        """ChatCompletionRequest 默认值。"""
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="测试")])
        assert req.model == "classical-chinese-llm"
        assert req.temperature == 0.7
        assert req.stream is False

    def test_chat_completion_request_validation(self) -> None:
        """messages 为空应验证失败。"""
        with pytest.raises(ValueError):
            ChatCompletionRequest(messages=[])


class TestChatAPI:
    """FastAPI 端点测试。"""

    @pytest.fixture
    def client(self) -> TestClient:
        """创建测试客户端。"""
        engine = _make_test_engine()
        app = create_app(engine)
        return TestClient(app)

    def test_health_check(self, client: TestClient) -> None:
        """健康检查端点。"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_non_stream_chat(self, client: TestClient) -> None:
        """非流式聊天请求。"""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "classical-chinese-llm",
                "messages": [
                    {"role": "system", "content": "你是文言文专家"},
                    {"role": "user", "content": "请解释天道"},
                ],
                "temperature": 0.7,
                "max_tokens": 50,
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "content" in data["choices"][0]["message"]

    def test_stream_chat(self, client: TestClient) -> None:
        """流式聊天请求。"""
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "classical-chinese-llm",
                "messages": [{"role": "user", "content": "测试"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200

            # 读取完整响应体
            body = response.read()
            assert len(body) > 0, "SSE 流响应不应为空"

            text = body.decode("utf-8")
            # SSE 格式: "data: {...}\n\n"
            assert "data: " in text
            assert "chat.completion.chunk" in text

    def test_missing_messages_field_returns_422(self, client: TestClient) -> None:
        """缺少 messages 字段应返回 422。"""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "classical-chinese-llm"},
        )
        assert response.status_code == 422

    def test_response_has_required_fields(self, client: TestClient) -> None:
        """响应应包含所有 OpenAI 兼容字段。"""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "测试"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        for field in ["id", "object", "created", "model", "choices"]:
            assert field in data, f"缺少字段: {field}"

    def test_service_info_endpoint(self, client: TestClient) -> None:
        """服务信息端点。"""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) >= 1
