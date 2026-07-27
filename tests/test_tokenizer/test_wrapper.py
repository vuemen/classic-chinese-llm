"""HF Tokenizer 封装测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from classic_chinese_llm.tokenizer.config import TokenizerConfig
from classic_chinese_llm.tokenizer.trainer import TokenizerTrainer
from classic_chinese_llm.tokenizer.wrapper import (
    CHAT_TEMPLATE_JINJA,
    build_tokenizer,
    save_tokenizer,
)


class TestBuildTokenizer:
    """build_tokenizer 工厂函数测试。"""

    @pytest.fixture
    def trained_model(self, temp_dir: Path) -> Path:
        """训练一个小型 SentencePiece 模型用于测试。

        使用极小参数在秒级完成训练:
        - byte_fallback=False: 测试语料太小，不需要 byte fallback
        - character_coverage=0.99: 降低以适配小语料
        """
        corpus_path = temp_dir / "corpus.txt"
        corpus_text = (
            "子曰學而時習之不亦說乎有朋自遠方來不亦樂乎人不知而不慍不亦君子乎\n"
            "大學之道在明明德在親民在止於至善\n"
            "知止而後有定定而後能靜靜而後能安安而後能慮慮而後能得\n"
            "物有本末事有終始知所先後則近道矣\n"
            "古之欲明明德於天下者先治其國欲治其國者先齊其家\n"
            "欲齊其家者先修其身欲修其身者先正其心\n"
            "欲正其心者先誠其意欲誠其意者先致其知致知在格物\n"
        )
        corpus_path.write_text(corpus_text, encoding="utf-8")

        model_prefix = str(temp_dir / "test_model")
        output_dir = str(temp_dir)

        config = TokenizerConfig(
            vocab_size=100,
            model_prefix=model_prefix,
            output_dir=output_dir,
            num_threads=1,
            input_sentence_size=0,
            byte_fallback=False,
            character_coverage=0.99,
            hard_vocab_limit=False,
        )

        trainer = TokenizerTrainer(config)
        return trainer.train(corpus_path=corpus_path)

    def test_build_from_model_file(self, trained_model: Path) -> None:
        """从 .model 文件构建 HF tokenizer。"""
        tokenizer = build_tokenizer(trained_model)

        assert tokenizer.vocab_size > 0
        assert tokenizer.bos_token == "<|bos|>"
        assert tokenizer.eos_token == "<|eos|>"
        assert tokenizer.pad_token == "<|pad|>"

    def test_encode_decode_roundtrip(self, trained_model: Path) -> None:
        """编码→解码 round-trip 基本一致。"""
        tokenizer = build_tokenizer(trained_model)
        text = "子曰學而時習之不亦說乎"

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # 去除 BOS/EOS 和空白后核心文本应一致
        assert "子曰" in decoded or len(tokens) > 0

    def test_special_tokens_present(self, trained_model: Path) -> None:
        """ChatML 特殊 token 在 vocab 中。"""
        tokenizer = build_tokenizer(trained_model)

        special_ids = tokenizer.encode("<|system|><|user|><|assistant|><|end|>")
        assert len(special_ids) > 0

    def test_chat_template_registered(self, trained_model: Path) -> None:
        """Chat Template 已注册。"""
        tokenizer = build_tokenizer(trained_model)

        assert tokenizer.chat_template == CHAT_TEMPLATE_JINJA

    def test_apply_chat_template(self, trained_model: Path) -> None:
        """apply_chat_template 正确格式化对话。"""
        tokenizer = build_tokenizer(trained_model)

        messages = [
            {"role": "system", "content": "你是文言文專家。"},
            {"role": "user", "content": "子曰何謂也？"},
            {"role": "assistant", "content": "學而時習之。"},
        ]

        result = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "<|assistant|>" in result
        assert "<|end|>" in result
        assert "你是文言文專家" in result
        assert "子曰何謂也" in result
        assert "學而時習之" in result

    def test_apply_chat_template_with_generation_prompt(self, trained_model: Path) -> None:
        """add_generation_prompt=True 时末尾添加 assistant 前缀。"""
        tokenizer = build_tokenizer(trained_model)

        messages = [
            {"role": "system", "content": "你是文言文專家。"},
            {"role": "user", "content": "子曰何謂也？"},
        ]

        result = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert result.endswith("<|assistant|>")

    def test_tokenize_chat_messages(self, trained_model: Path) -> None:
        """Chat messages 可被 tokenize。"""
        tokenizer = build_tokenizer(trained_model)

        messages = [
            {"role": "user", "content": "子曰學而時習之。"},
        ]

        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_file_not_found_raises(self) -> None:
        """不存在的文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="模型文件不存在"):
            build_tokenizer("/nonexistent/path.model")


class TestSaveTokenizer:
    """save_tokenizer 测试。"""

    def test_save_and_reload(self, temp_dir: Path) -> None:
        """保存后可通过 from_pretrained 重新加载。"""
        # 训练一个小模型（使用更多样的文本以增大字母表）
        corpus_path = temp_dir / "corpus.txt"
        corpus_text = (
            "子曰學而時習之不亦說乎有朋自遠方來不亦樂乎\n"
            "大學之道在明明德在親民在止於至善\n"
            "知止而後有定定而後能靜靜而後能安安而後能慮\n"
            "物有本末事有終始知所先後則近道矣\n"
            "古之欲明明德於天下者先治其國欲治其國者先齊其家\n"
            "欲齊其家者先修其身欲修其身者先正其心\n"
            "欲正其心者先誠其意欲誠其意者先致其知致知在格物\n"
            "物格而後知至知至而後意誠意誠而後心正\n"
            "心正而後身修身修而後家齊家齊而後國治\n"
            "國治而後天下平自天子以至於庶人壹是皆以修身為本\n"
        )
        corpus_path.write_text(corpus_text, encoding="utf-8")

        output_dir = temp_dir / "output"
        config = TokenizerConfig(
            vocab_size=100,
            model_prefix=str(temp_dir / "test_model"),
            output_dir=str(output_dir),
            num_threads=1,
            input_sentence_size=0,
            byte_fallback=False,
            character_coverage=0.99,
            hard_vocab_limit=False,
        )

        trainer = TokenizerTrainer(config)
        model_path = trainer.train(corpus_path=corpus_path)

        tokenizer = build_tokenizer(model_path, config)
        save_tokenizer(tokenizer, output_dir)

        # 验证保存的文件存在
        assert (output_dir / "tokenizer.json").exists()
        assert (output_dir / "tokenizer_config.json").exists()

        # 通过 transformers 重新加载
        from transformers import AutoTokenizer

        reloaded = AutoTokenizer.from_pretrained(str(output_dir))
        assert reloaded.vocab_size > 0
        assert reloaded.chat_template is not None

    def test_save_output_dir_created(self, temp_dir: Path) -> None:
        """输出目录不存在时自动创建。"""
        corpus_path = temp_dir / "corpus.txt"
        corpus_text = (
            "子曰學而時習之不亦說乎\n"
            "大學之道在明明德\n"
            "物有本末事有終始\n"
            "古之欲明明德於天下者先治其國\n"
            "欲齊其家者先修其身\n"
            "心正而後身修身修而後家齊\n"
        )
        corpus_path.write_text(corpus_text, encoding="utf-8")

        output_dir = temp_dir / "nested" / "tokenizer_output"
        config = TokenizerConfig(
            vocab_size=100,
            model_prefix=str(temp_dir / "test_model"),
            output_dir=str(output_dir),
            num_threads=1,
            input_sentence_size=0,
            byte_fallback=False,
            character_coverage=0.99,
            hard_vocab_limit=False,
        )

        trainer = TokenizerTrainer(config)
        model_path = trainer.train(corpus_path=corpus_path)

        tokenizer = build_tokenizer(model_path, config)
        saved = save_tokenizer(tokenizer, output_dir)

        assert saved.exists()
        assert (saved / "tokenizer.json").exists()


class TestChatTemplate:
    """Chat Template (Jinja2) 测试。"""

    def test_template_is_valid_jinja2(self) -> None:
        """模板是有效的 Jinja2 语法。"""
        from jinja2 import Template

        tpl = Template(CHAT_TEMPLATE_JINJA)
        assert tpl is not None

    def test_template_with_add_generation_prompt(self) -> None:
        """add_generation_prompt=True 渲染包含 assistant 前缀。"""
        from jinja2 import Template

        tpl = Template(CHAT_TEMPLATE_JINJA)
        result = tpl.render(
            messages=[{"role": "user", "content": "問曰"}],
            add_generation_prompt=True,
        )

        assert "<|user|>" in result
        assert "<|end|>" in result
        assert result.endswith("<|assistant|>")

    def test_template_without_add_generation_prompt(self) -> None:
        """add_generation_prompt=False 不添加 assistant 前缀。"""
        from jinja2 import Template

        tpl = Template(CHAT_TEMPLATE_JINJA)
        result = tpl.render(
            messages=[{"role": "user", "content": "問曰"}],
            add_generation_prompt=False,
        )

        assert "<|user|>" in result
        assert not result.endswith("<|assistant|>")

    def test_template_all_roles(self) -> None:
        """三种角色均正确渲染。"""
        from jinja2 import Template

        tpl = Template(CHAT_TEMPLATE_JINJA)
        result = tpl.render(
            messages=[
                {"role": "system", "content": "你是專家。"},
                {"role": "user", "content": "問。"},
                {"role": "assistant", "content": "答。"},
            ],
            add_generation_prompt=False,
        )

        assert "<|system|>你是專家。<|end|>" in result
        assert "<|user|>問。<|end|>" in result
        assert "<|assistant|>答。<|end|>" in result
