"""HF PreTrainedTokenizerFast 封装。

将训练好的 SentencePiece 模型封装为标准 PreTrainedTokenizerFast:
- 加载 SentencePiece .model 文件
- 配置文言文预分词器
- 注册 ChatML 特殊 token
- 内置 classical_chinese_v1 Chat Template
- 支持 save_pretrained / AutoTokenizer.from_pretrained
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer, models, normalizers, processors
from transformers import PreTrainedTokenizerFast

from classic_chinese_llm.tokenizer.config import TokenizerConfig
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# ─── Chat Template ──────────────────────────────────────────────────────

CHAT_TEMPLATE_JINJA = """\
{%- for message in messages %}
  {%- if message.role == 'system' %}
    {{- '<|system|>' + message.content + '<|end|>' }}
  {%- elif message.role == 'user' %}
    {{- '<|user|>' + message.content + '<|end|>' }}
  {%- elif message.role == 'assistant' %}
    {{- '<|assistant|>' + message.content + '<|end|>' }}
  {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
  {{- '<|assistant|>' }}
{%- endif %}"""


def build_tokenizer(
    model_path: str | Path,
    config: TokenizerConfig | None = None,
) -> PreTrainedTokenizerFast:
    """加载训练好的 SentencePiece 模型，封装为 PreTrainedTokenizerFast。

    Args:
        model_path: SentencePiece .model 文件路径。
        config: Tokenizer 配置。若为 None，使用默认配置。

    Returns:
        配置完成的 PreTrainedTokenizerFast 实例，可直接用于训练和推理。

    Raises:
        FileNotFoundError: .model 文件不存在时抛出。
    """
    if config is None:
        config = TokenizerConfig()

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    logger.info("加载 SentencePiece 模型: %s", model_path)

    # Step 1: 用 SentencePieceProcessor 读取 vocab
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    # 从 SentencePiece 模型提取 vocab 并构建 Unigram model
    vocab: list[tuple[str, float]] = []
    for idx in range(sp.vocab_size()):
        piece = sp.id_to_piece(idx)
        score = sp.get_score(idx)
        vocab.append((piece, score))

    unigram_model = models.Unigram(vocab, unk_id=config.unk_id)

    tokenizer = Tokenizer(unigram_model)
    # SentencePiece 模型内部已包含 normalizer 和 pre_tokenizer 的逻辑
    # Unity 规范化: NFKC 将全角/半角、异体字标准化
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])

    # Step 2: 设置后处理模板（添加 BOS/EOS）
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        pair=(
            f"{config.bos_token} $A {config.eos_token} " f"{config.bos_token} $B {config.eos_token}"
        ),
        special_tokens=[
            (config.bos_token, config.bos_id),
            (config.eos_token, config.eos_id),
        ],
    )

    # Step 3: 封装为 PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer,
        bos_token=config.bos_token,
        eos_token=config.eos_token,
        pad_token=config.pad_token,
        unk_token=config.unk_token,
        chat_template=CHAT_TEMPLATE_JINJA,
        model_max_length=2048,
    )

    # 手动添加 ChatML 特殊 token
    chatml_tokens = [
        config.system_token,
        config.user_token,
        config.assistant_token,
        config.end_token,
    ]
    hf_tokenizer.add_special_tokens({"additional_special_tokens": chatml_tokens})

    logger.info("HF Tokenizer 封装完成: vocab_size=%d", hf_tokenizer.vocab_size)
    return hf_tokenizer


def save_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    output_dir: str | Path,
) -> Path:
    """保存 tokenizer 到目录。

    生成以下文件:
        - tokenizer.json (Rust 序列化，Fast tokenizer 主文件)
        - tokenizer_config.json (元配置：特殊 token, chat_template)
        - special_tokens_map.json (特殊 token 名称 → ID 映射)
        - added_tokens.json (额外添加的 token)

    Args:
        tokenizer: 已封装的 HF tokenizer。
        output_dir: 输出目录（通常为 models/tokenizer/）。

    Returns:
        输出目录路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(str(output_dir))
    logger.info("Tokenizer 已保存至: %s", output_dir)
    return output_dir
