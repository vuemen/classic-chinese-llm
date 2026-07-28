"""训练数据集测试 —— PretrainDataset 与 SFTDataset。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from classic_chinese_llm.training.datasets import PretrainDataset, SFTDataset

# ═══════════════════════════════════════════════════════════════════════════
# Mock Tokenizers
# ═══════════════════════════════════════════════════════════════════════════


class MockPretrainTokenizer:
    """模拟 HuggingFace tokenizer，仅提供 PretrainDataset 所需的方法。"""

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 2048,
        return_tensors=None,
        **kwargs,
    ) -> dict[str, list[int]]:
        # 每个字符用其 Unicode 码位取模作为 token ID（范围 [10, 109]）
        ids = [(ord(c) % 100) + 10 for c in text]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


class MockSFTTokenizer:
    """模拟 HuggingFace tokenizer，提供 SFTDataset 所需的全部方法。

    特殊 token ID:
        pad=0, bos=1, eos=2, system=3, user=4, assistant=5, end=6
    内容 token ID 从 10 开始，确保不与特殊 token 冲突。
    """

    SPECIAL_TOKENS: dict[str, int] = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|system|>": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
        "<|end|>": 6,
    }

    def __init__(self) -> None:
        self._content_base = 10  # 内容 token ID 起点

    def _encode_content(self, text: str) -> list[int]:
        """将文本编码为 token ID 列表。"""
        return [(ord(c) % 100) + self._content_base for c in text]

    def convert_tokens_to_ids(self, token: str) -> int:
        """单个特殊 token 到 ID 的映射。"""
        return self.SPECIAL_TOKENS.get(token, -1)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        truncation: bool = True,
        max_length: int = 2048,
    ) -> list[int]:
        """模拟 ChatML 模板格式：
        <bos> <role_token> content... <end> ... <eos>
        """
        ids: list[int] = [self.SPECIAL_TOKENS["<|bos|>"]]
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                ids.append(self.SPECIAL_TOKENS["<|system|>"])
            elif role == "user":
                ids.append(self.SPECIAL_TOKENS["<|user|>"])
            elif role == "assistant":
                ids.append(self.SPECIAL_TOKENS["<|assistant|>"])
            ids.extend(self._encode_content(content))
            ids.append(self.SPECIAL_TOKENS["<|end|>"])
        ids.append(self.SPECIAL_TOKENS["<|eos|>"])
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        return ids


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def pretrain_tokenizer() -> MockPretrainTokenizer:
    """返回 PretrainDataset 所需的 mock tokenizer。"""
    return MockPretrainTokenizer()


@pytest.fixture
def sft_tokenizer() -> MockSFTTokenizer:
    """返回 SFTDataset 所需的 mock tokenizer。"""
    return MockSFTTokenizer()


@pytest.fixture
def pretrain_jsonl(temp_dir: Path) -> Path:
    """创建预训练 JSONL 数据文件，每行 {"text": "..."}。"""
    file_path = temp_dir / "pretrain.jsonl"
    samples = [
        {"text": "子曰学而时习之不亦说乎"},
        {"text": "有朋自远方来不亦乐乎"},
        {"text": "人不知而不愠不亦君子乎"},
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return file_path


@pytest.fixture
def pretrain_jsonl_with_skip(temp_dir: Path) -> Path:
    """包含空 text 和缺失 text 字段的 JSONL，应被正确跳过。"""
    file_path = temp_dir / "pretrain_skip.jsonl"
    lines = [
        {"text": "有效数据一"},
        {"text": ""},  # 空文本 → 跳过
        {"other": "无 text 字段"},  # 缺失 text → 跳过
        {"text": "有效数据二"},
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return file_path


@pytest.fixture
def sft_jsonl(temp_dir: Path) -> Path:
    """创建 SFT JSONL 数据文件，每行 {"messages": [...]}。"""
    file_path = temp_dir / "sft.jsonl"
    samples = [
        {
            "messages": [
                {"role": "system", "content": "你是一个翻译助手。"},
                {"role": "user", "content": "学而时习之"},
                {"role": "assistant", "content": "学而时习之"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "不亦说乎"},
                {"role": "assistant", "content": "不亦说乎"},
            ]
        },
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return file_path


@pytest.fixture
def sft_jsonl_multi_turn(temp_dir: Path) -> Path:
    """多轮对话的 SFT JSONL 数据。"""
    file_path = temp_dir / "sft_multi.jsonl"
    sample = {
        "messages": [
            {"role": "user", "content": "问"},
            {"role": "assistant", "content": "答一"},
            {"role": "user", "content": "再问"},
            {"role": "assistant", "content": "答二"},
        ]
    }
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return file_path


# ═══════════════════════════════════════════════════════════════════════════
# PretrainDataset Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPretrainDataset:
    """PretrainDataset 单元测试。"""

    def test_len_returns_correct_count(
        self, pretrain_jsonl: Path, pretrain_tokenizer: MockPretrainTokenizer
    ) -> None:
        """__len__ 返回有效样本行数。"""
        ds = PretrainDataset(pretrain_jsonl, pretrain_tokenizer)
        assert len(ds) == 3

    def test_skips_empty_and_missing_text(
        self, pretrain_jsonl_with_skip: Path, pretrain_tokenizer: MockPretrainTokenizer
    ) -> None:
        """空 text 和缺失 text 字段的行被跳过。"""
        ds = PretrainDataset(pretrain_jsonl_with_skip, pretrain_tokenizer)
        assert len(ds) == 2  # 仅 "有效数据一" 和 "有效数据二"

    def test_getitem_returns_input_ids_and_labels(
        self, pretrain_jsonl: Path, pretrain_tokenizer: MockPretrainTokenizer
    ) -> None:
        """__getitem__ 返回包含 input_ids 和 labels 的字典。"""
        ds = PretrainDataset(pretrain_jsonl, pretrain_tokenizer)
        item = ds[0]

        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long

    def test_labels_equal_input_ids(
        self, pretrain_jsonl: Path, pretrain_tokenizer: MockPretrainTokenizer
    ) -> None:
        """labels 是 input_ids 的副本（Causal LM 的 shift 由 attention mask 隐式处理）。"""
        ds = PretrainDataset(pretrain_jsonl, pretrain_tokenizer)
        item = ds[0]

        assert torch.equal(item["input_ids"], item["labels"])
        # 确保是独立副本（非同一 tensor）
        assert item["input_ids"] is not item["labels"]

    def test_getitem_different_samples_return_different_data(
        self, pretrain_jsonl: Path, pretrain_tokenizer: MockPretrainTokenizer
    ) -> None:
        """不同索引返回不同数据。"""
        ds = PretrainDataset(pretrain_jsonl, pretrain_tokenizer)
        item0 = ds[0]
        item1 = ds[1]

        # 不同样本的 input_ids 应不同（内容不同）
        assert not torch.equal(item0["input_ids"], item1["input_ids"])

    def test_max_seq_len_truncation(self, pretrain_tokenizer: MockPretrainTokenizer) -> None:
        """超过 max_seq_len 的序列被截断。"""
        # 构造一个长文本（> 5 个字符），设置 max_seq_len=5
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps({"text": "一二三四五六七八九十"}, ensure_ascii=False) + "\n")
            tmp_path = Path(f.name)

        try:
            ds = PretrainDataset(tmp_path, pretrain_tokenizer, max_seq_len=5)
            item = ds[0]
            assert item["input_ids"].shape[0] <= 5
        finally:
            tmp_path.unlink()

    def test_getitem_index_error(
        self, pretrain_jsonl: Path, pretrain_tokenizer: MockPretrainTokenizer
    ) -> None:
        """越界索引抛出 IndexError。"""
        ds = PretrainDataset(pretrain_jsonl, pretrain_tokenizer)
        with pytest.raises(IndexError):
            _ = ds[100]

    def test_one_dimensional_tensors(
        self, pretrain_jsonl: Path, pretrain_tokenizer: MockPretrainTokenizer
    ) -> None:
        """返回的 input_ids 和 labels 是 1D 张量。"""
        ds = PretrainDataset(pretrain_jsonl, pretrain_tokenizer)
        item = ds[0]
        assert item["input_ids"].dim() == 1
        assert item["labels"].dim() == 1


# ═══════════════════════════════════════════════════════════════════════════
# SFTDataset Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSFTDataset:
    """SFTDataset 单元测试。"""

    def test_len_returns_correct_count(
        self, sft_jsonl: Path, sft_tokenizer: MockSFTTokenizer
    ) -> None:
        """__len__ 返回有效样本行数。"""
        ds = SFTDataset(sft_jsonl, sft_tokenizer)
        assert len(ds) == 2

    def test_getitem_returns_input_ids_and_labels(
        self, sft_jsonl: Path, sft_tokenizer: MockSFTTokenizer
    ) -> None:
        """__getitem__ 返回 input_ids 和 labels 张量。"""
        ds = SFTDataset(sft_jsonl, sft_tokenizer)
        item = ds[0]

        assert "input_ids" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long
        assert item["input_ids"].dim() == 1
        assert item["labels"].dim() == 1

    def test_labels_ignore_system_and_user_tokens(
        self, sft_jsonl: Path, sft_tokenizer: MockSFTTokenizer
    ) -> None:
        """system 和 user token 位置在 labels 中为 -100。"""
        ds = SFTDataset(sft_jsonl, sft_tokenizer)
        item = ds[0]
        input_ids = item["input_ids"]
        labels = item["labels"]

        sp = MockSFTTokenizer.SPECIAL_TOKENS
        # 找到所有特殊 token 的位置
        sys_positions = (input_ids == sp["<|system|>"]).nonzero(as_tuple=True)[0]
        user_positions = (input_ids == sp["<|user|>"]).nonzero(as_tuple=True)[0]
        asst_positions = (input_ids == sp["<|assistant|>"]).nonzero(as_tuple=True)[0]
        end_positions = (input_ids == sp["<|end|>"]).nonzero(as_tuple=True)[0]
        bos_positions = (input_ids == sp["<|bos|>"]).nonzero(as_tuple=True)[0]
        eos_positions = (input_ids == sp["<|eos|>"]).nonzero(as_tuple=True)[0]

        # <bos> → -100
        for p in bos_positions:
            assert labels[p].item() == -100

        # <system> → -100
        for p in sys_positions:
            assert labels[p].item() == -100

        # <user> → -100
        for p in user_positions:
            assert labels[p].item() == -100

        # <assistant> 自身 → -100（不预测 assistant token 本身）
        for p in asst_positions:
            assert labels[p].item() == -100

        # <eos> → -100（在 assistant 的 <end> 之后）
        for p in eos_positions:
            assert labels[p].item() == -100

        # <end> 标记：只有 assistant 段的 <end> 保留，其余为 -100
        # 第一个 <end> 是 system 段结尾，第二个 <end> 是 user 段结尾
        # 最后一个 <end> 是 assistant 段结尾
        end_list = end_positions.tolist()
        assert len(end_list) >= 3  # sys end, user end, asst end
        # system 和 user 段的 <end> → -100
        for ep in end_list[:-1]:
            assert labels[ep].item() == -100
        # assistant 段的 <end> → 保留（值为对应 token ID）
        asst_end_pos = end_list[-1]
        assert labels[asst_end_pos].item() == sp["<|end|>"]

    def test_assistant_content_retained(
        self, sft_jsonl: Path, sft_tokenizer: MockSFTTokenizer
    ) -> None:
        """assistant 回复内容在 labels 中保留原始 token ID。"""
        ds = SFTDataset(sft_jsonl, sft_tokenizer)
        item = ds[0]
        input_ids = item["input_ids"]
        labels = item["labels"]

        sp = MockSFTTokenizer.SPECIAL_TOKENS
        asst_positions = (input_ids == sp["<|assistant|>"]).nonzero(as_tuple=True)[0]
        assert len(asst_positions) == 1
        asst_pos = asst_positions[0].item()

        # assistant token 之后、<end> 之前的 token 应保留
        end_after_asst = (input_ids[asst_pos + 1 :] == sp["<|end|>"]).nonzero(as_tuple=True)[0]
        assert len(end_after_asst) > 0
        asst_content_end = asst_pos + 1 + end_after_asst[0].item()

        for i in range(asst_pos + 1, asst_content_end + 1):
            assert (
                labels[i].item() == input_ids[i].item()
            ), f"位置 {i}: 期望 label={input_ids[i].item()}, 实际={labels[i].item()}"

    def test_multi_turn_all_assistant_retained(
        self, sft_jsonl_multi_turn: Path, sft_tokenizer: MockSFTTokenizer
    ) -> None:
        """多轮对话中每个 assistant 段的内容均被保留。"""
        ds = SFTDataset(sft_jsonl_multi_turn, sft_tokenizer)
        item = ds[0]
        input_ids = item["input_ids"]
        labels = item["labels"]

        sp = MockSFTTokenizer.SPECIAL_TOKENS
        asst_positions = (input_ids == sp["<|assistant|>"]).nonzero(as_tuple=True)[0]
        assert len(asst_positions) == 2, f"应有 2 个 assistant 段，实际 {len(asst_positions)}"

        for asst_idx_tensor in asst_positions:
            asst_pos = asst_idx_tensor.item()
            # assistant token 自身 → -100
            assert labels[asst_pos].item() == -100

            # 找该 assistant 的 <end>
            end_offsets = (input_ids[asst_pos + 1 :] == sp["<|end|>"]).nonzero(as_tuple=True)[0]
            assert len(end_offsets) > 0
            content_end = asst_pos + 1 + end_offsets[0].item()

            # assistant 内容区域 → 保留
            for i in range(asst_pos + 1, content_end + 1):
                assert labels[i].item() == input_ids[i].item()

            # <end> 之后的下一个 token（<eos> 或 <user>）→ -100
            if content_end + 1 < len(labels):
                assert labels[content_end + 1].item() == -100

    def test_no_assistant_all_ignored(
        self, temp_dir: Path, sft_tokenizer: MockSFTTokenizer
    ) -> None:
        """没有 assistant 回复时，所有 labels 为 -100。"""
        file_path = temp_dir / "no_asst.jsonl"
        sample = {
            "messages": [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "用户问题"},
            ]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        ds = SFTDataset(file_path, sft_tokenizer)
        item = ds[0]
        labels = item["labels"]

        assert (labels == -100).all()

    def test_custom_chat_template_accepted(
        self, sft_jsonl: Path, sft_tokenizer: MockSFTTokenizer
    ) -> None:
        """构造时传入自定义 chat_template 不报错。"""
        # chat_template 由 tokenizer 内部管理，SFTDataset 仅透传参数
        ds = SFTDataset(sft_jsonl, sft_tokenizer, chat_template="my_custom_template")
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item

    def test_skips_empty_messages(self, temp_dir: Path, sft_tokenizer: MockSFTTokenizer) -> None:
        """空 messages 列表的行被跳过。"""
        file_path = temp_dir / "skip_empty.jsonl"
        lines = [
            {
                "messages": [
                    {"role": "user", "content": "有效"},
                    {"role": "assistant", "content": "有效"},
                ]
            },
            {"messages": []},  # 空 → 跳过
            {"other": "no messages"},  # 缺失 → 跳过
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        ds = SFTDataset(file_path, sft_tokenizer)
        assert len(ds) == 1

    def test_getitem_index_error(self, sft_jsonl: Path, sft_tokenizer: MockSFTTokenizer) -> None:
        """越界索引抛出 IndexError。"""
        ds = SFTDataset(sft_jsonl, sft_tokenizer)
        with pytest.raises(IndexError):
            _ = ds[100]
