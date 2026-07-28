"""对话管理 —— 多轮对话历史维护与上下文截断。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    """单条对话消息。

    Attributes:
        role: 角色 ("system", "user", "assistant")。
        content: 消息文本内容。
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """转换为 {"role": ..., "content": ...} 字典格式。"""
        return {"role": self.role, "content": self.content}


class ConversationManager:
    """多轮对话管理器。

    职责:
    1. 维护消息历史 (system prompt + user/assistant 交替)
    2. 自动截断超长上下文（保留 system + 最近的 N 轮）
    3. 提供 messages 格式输出，可直接传给 InferenceEngine

    Args:
        system_prompt: 系统提示词（空字符串表示无 system prompt）。
        max_turns: 保留的最大对话轮次（1 轮 = user + assistant）。
    """

    def __init__(
        self,
        system_prompt: str = "",
        max_turns: int = 20,
    ) -> None:
        self.max_turns = max_turns
        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

    def add_user(self, content: str) -> None:
        """添加用户消息。

        Args:
            content: 用户输入文本。
        """
        self._messages.append(Message(role="user", content=content))
        self._truncate_if_needed()

    def add_assistant(self, content: str) -> None:
        """添加助手回复。

        Args:
            content: 助手生成的文本。
        """
        self._messages.append(Message(role="assistant", content=content))
        self._truncate_if_needed()

    def get_messages(self) -> list[dict[str, str]]:
        """获取当前消息历史（ChatML 格式）。

        Returns:
            list[dict]: [{"role": "...", "content": "..."}, ...]。
        """
        return [msg.to_dict() for msg in self._messages]

    def reset(self) -> None:
        """清空对话历史（保留 system prompt）。"""
        system_msg = self._messages[0] if self._has_system() else None
        self._messages = []
        if system_msg is not None:
            self._messages.append(system_msg)

    def set_system_prompt(self, prompt: str) -> None:
        """修改 system prompt。

        Args:
            prompt: 新的系统提示词文本。
        """
        if self._has_system():
            self._messages[0] = Message(role="system", content=prompt)
        elif prompt:
            self._messages.insert(0, Message(role="system", content=prompt))

    # ─── 内部方法 ────────────────────────────────────────────────────

    def _has_system(self) -> bool:
        """检查是否有 system prompt。"""
        return len(self._messages) > 0 and self._messages[0].role == "system"

    def _truncate_if_needed(self) -> None:
        """如果对话轮次超出 max_turns，移除最早的非 system 消息。"""
        has_sys = self._has_system()
        # 仅计算 user + assistant 轮次
        turn_count = (len(self._messages) - (1 if has_sys else 0)) // 2

        while turn_count > self.max_turns:
            # 移除最早的非 system 消息的 user 和 assistant
            start_idx = 1 if has_sys else 0
            if start_idx + 1 < len(self._messages):
                del self._messages[start_idx : start_idx + 2]
            else:
                break
            turn_count -= 1
