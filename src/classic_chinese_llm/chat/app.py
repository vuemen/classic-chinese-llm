"""Gradio Web UI —— 文言文风格对话界面。

使用 Gradio Blocks 构建，支持:
- 参数调节面板 (temperature, top-p, top-k, max_tokens)
- 系统提示词选择
- 多轮对话
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from classic_chinese_llm.chat.prompts import list_system_prompts
from classic_chinese_llm.inference.engine import InferenceEngine


def create_ui(engine: InferenceEngine) -> gr.Blocks:
    """创建 Gradio Blocks 聊天界面。

    Args:
        engine: 推理引擎实例。

    Returns:
        gr.Blocks: Gradio 应用。
    """
    prompt_names = list_system_prompts()

    with gr.Blocks(
        title="文言文 LLM 对话",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("""
            # \U0001f4dc 文言文大语言模型对话系统
            与 AI 用文言文进行对话交流。支持古文问答、诗词创作、历史讲述等多种模式。
            """)

        with gr.Row():
            # 左侧: 参数面板
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### ⚙️ 参数设置")

                system_prompt_dd = gr.Dropdown(
                    choices=prompt_names,
                    value=prompt_names[-1] if prompt_names else "",
                    label="系统提示词角色",
                    interactive=True,
                )

                temperature_sl = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.05,
                    label="Temperature",
                    info="越高越随机，越低确定性越强",
                )

                top_p_sl = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    label="Top-P (Nucleus)",
                    info="累积概率阈值",
                )

                top_k_sl = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=5,
                    label="Top-K",
                    info="0 = 不使用",
                )

                max_tokens_sl = gr.Slider(
                    minimum=16,
                    maximum=1024,
                    value=256,
                    step=16,
                    label="最大生成长度",
                    info="生成的 token 上限",
                )

                clear_btn = gr.Button("\U0001f5d1️ 清空对话", variant="secondary", size="sm")

            # 右侧: 聊天区域
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=500,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="请用文言文或白话文输入您的问题...",
                        label="输入",
                        scale=5,
                        show_label=False,
                    )
                    send_btn = gr.Button("↑ 发送", variant="primary", scale=1)

        # ── 事件处理 ──────────────────────────────────────────────────

        async def on_send(
            message: str,
            history: list[list[str | None]],
            system_prompt: str,
            temperature: float,
            top_p: float,
            top_k: int,
            max_tokens: int,
        ) -> Any:
            """处理用户发送消息事件。"""
            if not message.strip():
                return "", history

            # 构建对话历史（用于引擎）
            chat_history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
            for h in history:
                if h[0] is not None:
                    chat_history.append({"role": "user", "content": str(h[0])})
                if h[1] is not None:
                    chat_history.append({"role": "assistant", "content": str(h[1])})

            # 使用引擎生成
            generated = engine.generate(
                prompt=message,
                history=chat_history,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_tokens,
            )

            # 更新 Gradio chatbot history（格式: list of [user_msg, assistant_msg]）
            history.append([message, generated])
            return "", history

        def on_clear() -> list[list[str | None]]:
            """清空对话历史。"""
            return []

        # 绑定事件
        send_btn.click(
            on_send,
            inputs=[
                msg_input,
                chatbot,
                system_prompt_dd,
                temperature_sl,
                top_p_sl,
                top_k_sl,
                max_tokens_sl,
            ],
            outputs=[msg_input, chatbot],
        )

        msg_input.submit(
            on_send,
            inputs=[
                msg_input,
                chatbot,
                system_prompt_dd,
                temperature_sl,
                top_p_sl,
                top_k_sl,
                max_tokens_sl,
            ],
            outputs=[msg_input, chatbot],
        )

        clear_btn.click(
            on_clear,
            outputs=[chatbot],
        )

    return app  # type: ignore[no-any-return]
