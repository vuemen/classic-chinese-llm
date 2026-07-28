# 推理服务层设计文档

**所属阶段:** Phase 6 — 对话界面
**涉及模块:** `src/classic_chinese_llm/inference/`
**日期:** 2026-07-28

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | 模型加载 | 从 checkpoint 加载模型权重 + tokenizer，自动检测设备并将模型加载到 GPU/CPU |
| F2 | 文本生成 | 封装 Generator，提供高层 `generate()` 接口（接受文本输入返回文本输出） |
| F3 | 流式输出 | 提供 `stream()` 接口，逐 token yield 文本段，适用于 SSE/WebSocket |
| F4 | KV Cache 推理 | 使用 KV Cache 加速逐 token 推理（非流式也要支持） |
| F5 | Chat Template | 使用 tokenizer 的 chat template 格式化多轮对话 |

### 1.2 非功能需求

- 推理引擎与上层（Gradio/FastAPI）解耦，仅依赖 model 和 tokenizer 模块
- 支持 CPU 推理（无 GPU 环境也可运行，速度较慢但可用）
- 加载失败时提供明确的错误信息（checkpoint 不存在、格式不匹配等）

---

## 2. 组件设计

### 2.1 InferenceEngine

位置: `src/classic_chinese_llm/inference/engine.py`

```python
class InferenceEngine:
    """推理引擎 —— 高层封装。

    职责:
    1. 加载 checkpoint + tokenizer
    2. 提供 generate() 和 stream() 接口
    3. 管理设备、dtype、KV Cache

    Args:
        model: 已加载的 TransformerLM 模型
        tokenizer: PreTrainedTokenizerFast 实例
        generation_config: 默认生成参数
    """

    def __init__(self, model, tokenizer, generation_config=None): ...

    def generate(self, prompt, history=None, **kwargs) -> str:
        """非流式生成。"""

    def stream(self, prompt, history=None, **kwargs) -> Iterator[str]:
        """流式生成，逐 token yield。"""

    @classmethod
    def from_checkpoint(cls, checkpoint_path, tokenizer, config=None, device=None) -> InferenceEngine:
        """工厂方法: 从 checkpoint 加载模型并创建引擎。"""
```

### 2.2 接口简化

- `generate(prompt)` 接收纯文本 prompt，内部处理 tokenize → generate → decode
- `stream(prompt)` 返回 Python Generator，每次 yield 一个新 token 的文本
- 支持 `history` 参数：`list[dict[str,str]]` 格式的消息历史，调用 `apply_chat_template` 构建输入

---

## 3. 模块结构

```
src/classic_chinese_llm/inference/
├── __init__.py   # 导出 InferenceEngine
└── engine.py     # InferenceEngine 实现

tests/test_inference/
├── __init__.py
└── test_engine.py
```

---

## 4. 验证清单

- [ ] 从 checkpoint 加载成功
- [ ] generate 返回非空文本
- [ ] stream 至少 yield 1 个 token
- [ ] CPU 上可运行
- [ ] history 参数正常工作（多轮对话格式）
- [ ] checkpoint 不存在时抛出 FileNotFoundError
