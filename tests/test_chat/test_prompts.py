"""chat.prompts 模块的单元测试。"""

from __future__ import annotations

import pytest

from classic_chinese_llm.chat.prompts import (
    SYSTEM_PROMPTS,
    get_system_prompt,
    list_system_prompts,
)


class TestSystemPrompts:
    """系统提示词测试。"""

    def test_system_prompts_is_dict(self) -> None:
        """SYSTEM_PROMPTS 应为非空字典。"""
        assert isinstance(SYSTEM_PROMPTS, dict)
        assert len(SYSTEM_PROMPTS) >= 4

    def test_known_prompts_exist(self) -> None:
        """预定义的角色应存在。"""
        expected_roles = ["古文专家", "诗词创作", "历史讲述", "文言翻译"]
        for role in expected_roles:
            assert role in SYSTEM_PROMPTS, f"缺少角色: {role}"

    def test_get_system_prompt_valid(self) -> None:
        """获取已知角色应返回非空字符串。"""
        prompt = get_system_prompt("古文专家")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_system_prompt_invalid_raises(self) -> None:
        """未知角色应抛出 KeyError。"""
        with pytest.raises(KeyError):
            get_system_prompt("不存在的角色")

    def test_list_system_prompts(self) -> None:
        """list_system_prompts 应返回与 SYSTEM_PROMPTS 一致的列表。"""
        names = list_system_prompts()
        assert isinstance(names, list)
        assert len(names) == len(SYSTEM_PROMPTS)
        for name in names:
            assert name in SYSTEM_PROMPTS

    def test_all_prompts_non_empty(self) -> None:
        """所有 prompt 应为非空字符串。"""
        for name, prompt in SYSTEM_PROMPTS.items():
            assert len(prompt) > 0, f"角色 '{name}' 的 prompt 为空"
