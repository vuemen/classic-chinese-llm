"""chat.conversation 模块的单元测试。"""

from __future__ import annotations

from classic_chinese_llm.chat.conversation import ConversationManager, Message


class TestMessage:
    """Message 数据类测试。"""

    def test_creation(self) -> None:
        """基本创建。"""
        msg = Message(role="user", content="你好")
        assert msg.role == "user"
        assert msg.content == "你好"

    def test_to_dict(self) -> None:
        """转换为 dict 格式。"""
        msg = Message(role="assistant", content="回复内容")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "回复内容"}


class TestConversationManager:
    """ConversationManager 测试。"""

    def test_initial_state(self) -> None:
        """初始状态: 无用户消息。"""
        cm = ConversationManager()
        assert len(cm.get_messages()) == 0

    def test_initial_system_prompt(self) -> None:
        """初始化时可设置 system prompt。"""
        cm = ConversationManager(system_prompt="你是文言文助手")
        msgs = cm.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_add_user_and_assistant(self) -> None:
        """添加对话轮次。"""
        cm = ConversationManager()
        cm.add_user("问题一")
        msgs = cm.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

        cm.add_assistant("回答一")
        msgs = cm.get_messages()
        assert len(msgs) == 2
        assert msgs[1]["role"] == "assistant"

    def test_multiple_turns(self) -> None:
        """多轮对话。"""
        cm = ConversationManager()
        for i in range(3):
            cm.add_user(f"问题{i}")
            cm.add_assistant(f"回答{i}")
        msgs = cm.get_messages()
        assert len(msgs) == 6
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_reset_clears_history(self) -> None:
        """reset 应清空历史但保留 system prompt。"""
        cm = ConversationManager(system_prompt="你是助手")
        cm.add_user("问题")
        cm.add_assistant("回答")
        cm.reset()
        msgs = cm.get_messages()
        # 重置后只保留 system prompt
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_reset_no_system_prompt(self) -> None:
        """reset 无 system prompt 时应清空所有。"""
        cm = ConversationManager()
        cm.add_user("问题")
        cm.add_assistant("回答")
        cm.reset()
        assert len(cm.get_messages()) == 0

    def test_set_system_prompt(self) -> None:
        """动态修改 system prompt。"""
        cm = ConversationManager(system_prompt="旧提示")
        cm.set_system_prompt("新提示")
        msgs = cm.get_messages()
        assert msgs[0]["content"] == "新提示"

    def test_max_turns_truncation(self) -> None:
        """超出 max_turns 时自动截断。"""
        cm = ConversationManager(max_turns=2)
        for i in range(5):
            cm.add_user(f"问题{i}")
            cm.add_assistant(f"回答{i}")
        msgs = cm.get_messages()
        # system(0) + 2 turns(4 messages) = 4-5 条消息
        assert len(msgs) <= 5

    def test_truncation_preserves_system(self) -> None:
        """截断时 system prompt 始终保留。"""
        cm = ConversationManager(system_prompt="你是助手", max_turns=1)
        for i in range(5):
            cm.add_user(f"问题{i}")
            cm.add_assistant(f"回答{i}")
        msgs = cm.get_messages()
        assert msgs[0]["role"] == "system"
        # 仅保留最近的 1 轮 (1 user + 1 assistant) + system
        assert len(msgs) == 3
