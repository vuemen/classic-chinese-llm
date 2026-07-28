"""训练数据集 —— PretrainDataset 和 SFTDataset。

PretrainDataset: 从清洗后的 JSONL 加载原始文言文文本。
SFTDataset: 从 ChatML 格式的 JSONL 加载指令数据, 构建仅 assistant 的 labels。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class PretrainDataset(Dataset):  # type: ignore[type-arg]
    """预训练数据集。

    从 deduplicated.jsonl 逐行读取文言文文本, tokenize 后返回。

    Args:
        data_path: JSONL 数据文件路径。
        tokenizer: HF PreTrainedTokenizerFast 实例。
        max_seq_len: 最大序列长度 (截断)。
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._samples: list[str] = []

        data_path = Path(data_path)
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record.get("text", "").strip()
                if text:
                    self._samples.append(text)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self._samples[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors=None,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        # labels = input_ids (Causal LM 的 shift 由 CrossEntropyLoss 隐式处理)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class SFTDataset(Dataset):  # type: ignore[type-arg]
    """SFT 指令微调数据集。

    从 ChatML 格式的 JSONL 加载数据, 使用 tokenizer.apply_chat_template()
    将 messages 转换为 input_ids, 并构建仅 assistant 位置的 labels。

    数据格式 (每行):
        {"messages": [{"role": "system", "content": ...},
                       {"role": "user", "content": ...},
                       {"role": "assistant", "content": ...}], "task_type": "..."}

    Args:
        data_path: ChatML JSONL 数据文件路径。
        tokenizer: HF PreTrainedTokenizerFast 实例。
        max_seq_len: 最大序列长度。
        chat_template: Chat template 名称 (默认 "classical_chinese_v1")。
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 2048,
        chat_template: str = "classical_chinese_v1",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 获取 ChatML 特殊 token ID
        self.assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
        self.end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")

        # 加载样本
        self._samples: list[list[dict[str, str]]] = []
        data_path = Path(data_path)
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                messages = record.get("messages", [])
                if messages:
                    self._samples.append(messages)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self._samples[idx]

        # 使用 Chat Template 格式化 + Tokenize
        # add_generation_prompt=False: 训练时不追加 <|assistant|>
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.max_seq_len,
        )
        # tokenize=True 时返回 list[int]
        assert isinstance(tokenized, list), "apply_chat_template 应返回 list[int]"

        input_ids = torch.tensor(tokenized, dtype=torch.long)

        # convert_tokens_to_ids 对单个 token 返回 int
        asst_id = self.assistant_token_id
        end_id = self.end_token_id
        assert isinstance(asst_id, int), "assistant_token_id 类型错误"
        assert isinstance(end_id, int), "end_token_id 类型错误"

        # 构建仅 assistant 位置的 labels
        labels = _build_sft_labels(
            input_ids,
            assistant_token_id=asst_id,
            end_token_id=end_id,
        )

        return {"input_ids": input_ids, "labels": labels}


def _build_sft_labels(
    input_ids: torch.Tensor,
    assistant_token_id: int,
    end_token_id: int,
) -> torch.Tensor:
    """为 SFT 构建 label mask。

    算法:
    1. 扫描序列找到所有 <|assistant|> token 的位置
    2. 从 <|assistant|>+1 到 <|end|> 之间的 token 保留 label
    3. 其余位置设为 -100 (CrossEntropyLoss 的 ignore_index)

    注意: CrossEntropyLoss 内部做 predict[t]→target[t] 的隐式 shift,
    所以 <|assistant|> 本身不需要被预测, 但 <|assistant|>+1 开始的内容需要。

    ChatML 示例:
        <|bos|> <|sys|> ... <|end|> <|user|> ... <|end|> <|asst|> A <|end|> <|eos|>
        Label:   -100   -100  ...  -100    -100  ...  -100    -100   A <|end|> <|eos|>
    """
    labels = input_ids.clone()

    # 找到所有 <|assistant|> 的位置
    assistant_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]

    # 默认所有位置设为忽略
    labels[:] = -100

    for asst_pos in assistant_positions:
        pos = asst_pos.item()
        # 找到对应的 <|end|>
        end_positions = (input_ids[pos + 1 :] == end_token_id).nonzero(as_tuple=True)[0]
        if len(end_positions) == 0:
            # 截断的序列: assistant 之后全部保留
            labels[pos + 1 :] = input_ids[pos + 1 :]
        else:
            end_pos = pos + 1 + end_positions[0].item()
            # 保留 asst_pos+1 到 end_pos (含 <|end|>)
            labels[pos + 1 : end_pos + 1] = input_ids[pos + 1 : end_pos + 1]

    return labels
