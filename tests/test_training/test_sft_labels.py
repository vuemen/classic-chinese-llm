"""SFT label masking 测试 —— _build_sft_labels 函数。"""

from __future__ import annotations

import pytest
import torch

from classic_chinese_llm.training.datasets import _build_sft_labels


class TestBuildSFTLabels:
    """_build_sft_labels 单元测试。"""

    @pytest.fixture
    def token_ids(self) -> dict[str, int]:
        """模拟 token ID 映射。"""
        return {
            "bos": 2,
            "system": 4,
            "user": 5,
            "assistant": 6,
            "end": 7,
            "eos": 3,
            "pad": 0,
        }

    def test_only_assistant_retained(self, token_ids: dict[str, int]) -> None:
        """仅 assistant 位置的 label 被保留。"""
        # <bos> <sys> S1 <end> <user> U1 <end> <asst> A1 A2 <end> <eos>
        ids = token_ids
        input_ids = torch.tensor(
            [
                ids["bos"],
                ids["system"],
                10,
                ids["end"],
                ids["user"],
                20,
                ids["end"],
                ids["assistant"],
                30,
                40,
                ids["end"],
                ids["eos"],
            ]
        )

        labels = _build_sft_labels(
            input_ids,
            assistant_token_id=ids["assistant"],
            end_token_id=ids["end"],
        )

        # system / user / special tokens → -100
        assert (labels[0:1] == -100).all()  # bos
        assert (labels[1:3] == -100).all()  # <sys> S1
        assert labels[3] == -100  # <end>
        assert (labels[4:6] == -100).all()  # <user> U1
        assert labels[6] == -100  # <end>
        assert labels[7] == -100  # <assistant> (不预测)

        # assistant 内容保留
        assert labels[8] == 30  # A1
        assert labels[9] == 40  # A2
        assert labels[10] == ids["end"]  # <end> 需要预测（assistant 段内）
        # eos 在 assistant 的 <end> 之后，不属于 assistant 段，应忽略
        assert labels[11] == -100

    def test_multi_turn_conversation(self, token_ids: dict[str, int]) -> None:
        """多轮对话：每个 assistant 段独立保留。"""
        ids = token_ids
        # <user> Q1 <end> <asst> A1 <end> <user> Q2 <end> <asst> A2 <end>
        input_ids = torch.tensor(
            [
                ids["user"],
                11,
                ids["end"],
                ids["assistant"],
                21,
                ids["end"],
                ids["user"],
                12,
                ids["end"],
                ids["assistant"],
                22,
                ids["end"],
            ]
        )

        labels = _build_sft_labels(
            input_ids,
            assistant_token_id=ids["assistant"],
            end_token_id=ids["end"],
        )

        # Q1 → -100
        assert (labels[:3] == -100).all()
        # A1 assistant 不在标签中（不预测）
        assert labels[3] == -100  # <asst>
        # A1 内容 → 保留
        assert labels[4] == 21
        assert labels[5] == ids["end"]
        # Q2 → -100
        assert (labels[6:8] == -100).all()
        assert labels[8] == -100  # <end> of Q2
        # A2 assistant 不在标签中（不预测）
        assert labels[9] == -100  # <asst>
        # A2 内容 → 保留
        assert labels[10] == 22
        assert labels[11] == ids["end"]

    def test_truncated_no_end(self, token_ids: dict[str, int]) -> None:
        """没有 <end> 的截断序列：assistant 之后全部保留。"""
        ids = token_ids
        # <asst> A1 A2（被截断，没有 <end>）
        input_ids = torch.tensor([ids["assistant"], 30, 40])

        labels = _build_sft_labels(
            input_ids,
            assistant_token_id=ids["assistant"],
            end_token_id=ids["end"],
        )

        assert labels[0] == -100  # <assistant> 不预测
        assert labels[1] == 30  # A1 → 保留
        assert labels[2] == 40  # A2 → 保留

    def test_no_assistant_all_ignored(self, token_ids: dict[str, int]) -> None:
        """没有 <assistant> token 时全部为 -100。"""
        ids = token_ids
        input_ids = torch.tensor([ids["system"], 10, ids["end"], ids["user"], 20, ids["end"]])

        labels = _build_sft_labels(
            input_ids,
            assistant_token_id=ids["assistant"],
            end_token_id=ids["end"],
        )

        assert (labels == -100).all()
