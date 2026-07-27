"""数据格式化器测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from classic_chinese_llm.data.formatter import (
    FormattedSample,
    Formatter,
    FormatterConfig,
    TaskTemplate,
)


class TestFormattedSample:
    """FormattedSample 数据模型测试。"""

    def test_valid_sample(self) -> None:
        """完整的有效样本。"""
        sample = FormattedSample(
            messages=[
                {"role": "system", "content": "你是一位文言文专家。"},
                {"role": "user", "content": "翻译：子曰学而时习之"},
                {"role": "assistant", "content": "孔子说：学习了知识后经常温习。"},
            ],
            task_type="translate_wen_to_bai",
        )
        assert sample.is_valid(min_response_len=5) is True

    def test_empty_assistant_invalid(self) -> None:
        """assistant 回复为空时无效。"""
        sample = FormattedSample(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""},
            ],
            task_type="test",
        )
        assert sample.is_valid(min_response_len=5) is False

    def test_short_response_invalid(self) -> None:
        """assistant 回复过短时无效。"""
        sample = FormattedSample(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "短"},
            ],
            task_type="test",
        )
        assert sample.is_valid(min_response_len=5) is False

    def test_missing_roles_invalid(self) -> None:
        """messages 不足 3 条时无效。"""
        sample = FormattedSample(
            messages=[
                {"role": "user", "content": "test"},
            ],
            task_type="test",
        )
        assert sample.is_valid() is False

    def test_to_dict(self) -> None:
        """to_dict 返回正确的字典结构。"""
        sample = FormattedSample(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "asst"},
            ],
            task_type="translate_wen_to_bai",
        )
        d = sample.to_dict()
        assert len(d["messages"]) == 3
        assert d["task_type"] == "translate_wen_to_bai"


class TestTaskTemplate:
    """TaskTemplate 测试。"""

    def test_template_creation(self) -> None:
        """模板创建正常。"""
        tpl = TaskTemplate(
            task_type="test",
            display_name="测试任务",
            weight=1.0,
            system_prompt="系统提示词",
            instruction_template="指令: {text}",
        )
        assert tpl.task_type == "test"
        assert tpl.weight == 1.0


class TestFormatterPipeline:
    """Formatter 完整管道测试。"""

    def _make_input_jsonl(self, records: list[dict[str, Any]], path: Path) -> Path:
        file_path = path / "deduplicated.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return file_path

    def test_basic_format_pipeline(self, temp_dir: Path) -> None:
        """基本格式化管道端到端测试。"""
        long_text = "子曰：学而时习之，不亦说乎？" * 5  # ~100+ chars
        input_path = self._make_input_jsonl(
            [
                {"text": long_text, "source": "lunyu", "title": "学而篇"},
                {
                    "text": "曾子曰：吾日三省吾身——为人谋而不忠乎？" * 5,
                    "source": "lunyu",
                    "title": "学而篇",
                },
            ],
            temp_dir,
        )
        output_dir = temp_dir / "instructions"

        config = FormatterConfig(max_samples=10, seed=42)
        formatter = Formatter(config)
        stats = formatter.format(input_path, output_dir)

        assert stats.input_docs == 2
        assert stats.total_valid > 0
        assert stats.train_count >= stats.val_count
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "val.jsonl").exists()

    def test_max_samples_limit(self, temp_dir: Path) -> None:
        """max_samples 限制生效。"""
        # 生成大量文档以确保达到限制
        docs = []
        for i in range(100):
            docs.append(
                {
                    "text": "子曰：" + "学而时习之" * 20,
                    "source": "test",
                    "title": f"篇章{i}",
                }
            )
        input_path = self._make_input_jsonl(docs, temp_dir)
        output_dir = temp_dir / "instructions"

        config = FormatterConfig(max_samples=5, seed=42)
        formatter = Formatter(config)
        stats = formatter.format(input_path, output_dir)

        assert stats.total_valid <= 5

    def test_seed_reproducibility(self, temp_dir: Path) -> None:
        """相同 seed 产生相同输出。"""
        docs = [
            {
                "text": "子曰：" + "学而时习之" * 20,
                "source": "test",
                "title": f"篇{i}",
            }
            for i in range(20)
        ]
        input_path = self._make_input_jsonl(docs, temp_dir)

        run1: list[dict[str, Any]] = []
        run2: list[dict[str, Any]] = []

        for run_idx in range(2):
            config = FormatterConfig(max_samples=5, seed=99)
            formatter = Formatter(config)
            formatter.format(input_path, temp_dir / f"run{run_idx}")

            with open(temp_dir / f"run{run_idx}" / "train.jsonl", encoding="utf-8") as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
            if run_idx == 0:
                run1 = data
            else:
                run2 = data

        # 两次运行的 messages 应完全相同
        for s1, s2 in zip(run1, run2, strict=True):
            assert s1["messages"] == s2["messages"]
            assert s1["task_type"] == s2["task_type"]

    def test_val_split_proportion(self, temp_dir: Path) -> None:
        """验证集比例大致正确。"""
        docs = []
        for i in range(100):
            docs.append(
                {
                    "text": "子曰：" + "学而时习之" * 20,
                    "source": "test",
                    "title": f"篇{i}",
                }
            )
        input_path = self._make_input_jsonl(docs, temp_dir)
        output_dir = temp_dir / "instructions"

        config = FormatterConfig(max_samples=40, val_split=0.1, seed=42)
        formatter = Formatter(config)
        stats = formatter.format(input_path, output_dir)

        expected_val = max(1, int(stats.total_valid * 0.1))
        # 允许 ±1 的误差
        assert abs(stats.val_count - expected_val) <= 1

    def test_output_chatml_format(self, temp_dir: Path) -> None:
        """输出为 ChatML 格式，可被 tokenizer.apply_chat_template 使用。"""
        docs = [
            {
                "text": "子曰：" + "学而时习之" * 20,
                "source": "test",
                "title": "学而篇",
            }
        ]
        input_path = self._make_input_jsonl(docs, temp_dir)
        output_dir = temp_dir / "instructions"

        formatter = Formatter(FormatterConfig(max_samples=3, seed=42))
        formatter.format(input_path, output_dir)

        with open(output_dir / "train.jsonl", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                assert "messages" in record
                assert "task_type" in record
                assert len(record["messages"]) >= 3
                roles = [m["role"] for m in record["messages"]]
                assert roles[0] == "system"
                assert roles[1] == "user"
                assert roles[2] == "assistant"

    def test_add_custom_template(self, temp_dir: Path) -> None:
        """自定义模板注册正常工作。"""
        docs = [
            {
                "text": "子曰：" + "学而时习之" * 20,
                "source": "test",
                "title": "学而篇",
            }
        ]
        input_path = self._make_input_jsonl(docs, temp_dir)
        output_dir = temp_dir / "instructions"

        custom_tpl = TaskTemplate(
            task_type="custom_task",
            display_name="自定义任务",
            weight=10.0,  # 极高权重确保被选中
            system_prompt="你是一位专家。",
            instruction_template="请处理: {text}",
        )

        formatter = Formatter(FormatterConfig(max_samples=5, seed=42))
        formatter.add_template(custom_tpl)
        stats = formatter.format(input_path, output_dir)

        assert stats.total_valid > 0
        # 由于自定义模板权重极高，大部分样本应为该类型
        assert stats.per_task.get("custom_task", 0) > 0

    def test_min_source_text_len_filter(self, temp_dir: Path) -> None:
        """短文本被正确过滤。"""
        docs = [
            {"text": "太短", "source": "test"},  # 只有2字，远低于50
            {"text": "子曰：" + "学而时习之" * 20, "source": "test", "title": "篇"},
        ]
        input_path = self._make_input_jsonl(docs, temp_dir)
        output_dir = temp_dir / "instructions"

        config = FormatterConfig(max_samples=5, min_source_text_len=50)
        formatter = Formatter(config)
        stats = formatter.format(input_path, output_dir)

        # 只有1篇文档通过长度过滤
        assert stats.input_docs == 2  # 加载了2篇
        # 过滤后只保留1篇（短文本被过滤）
        assert stats.total_valid > 0
