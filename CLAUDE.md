# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Classical Chinese (文言文) conversational LLM, built to learn Transformers from first principles. A single ~157M parameter GPT-2-style Decoder-only Transformer is implemented using only `torch.nn`, then pretrained and instruction-tuned on classical Chinese corpora.

- **Python:** 3.12
- **Hardware:** NVIDIA GPU with 12GB+ VRAM, BF16 support
- **Training time:** ~2-3 days (pretrain) + ~2-4 hours (SFT)
- **No HF model code** — Transformers (`transformers`, `peft`, `bitsandbytes`, `trl`) are NOT used for model construction

See [docs/architecture.md](docs/architecture.md) for the full architecture design.

## Common Commands

```bash
# Install (create a Python 3.12 venv/conda env first)
pip install -r requirements.txt
pip install -e .

# Code quality (required before commits)
black src/ tests/
ruff check src/ tests/
mypy src/

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_model/test_layers.py -v

# Data pipeline
python scripts/collect_data.py --raw-dir data/raw
python scripts/train_tokenizer.py --corpus data/processed/cleaned.txt --vocab-size 32000

# Training
python scripts/pretrain.py --config configs/pretrain.yaml    # ~2-3 days
python scripts/finetune.py --config configs/sft.yaml          # ~2-4 hours

# Inference
python scripts/chat.py --checkpoint models/checkpoints/sft_best.pt
python scripts/serve.py --checkpoint models/checkpoints/sft_best.pt --port 8000
```

## Project Structure

```
src/classic_chinese_llm/
├── config/         # Pydantic settings (settings.py), path constants (paths.py)
├── data/           # sources/, collector, cleaner, deduplicator, formatter
├── tokenizer/      # SentencePiece Unigram trainer + pre-tokenizer
├── model/          # layers.py, config.py, transformer.py, generation.py
├── training/       # trainer.py, pretrain.py, sft.py, callbacks.py, data_collator.py
├── evaluation/     # metrics.py, evaluator.py
├── inference/      # engine.py (load checkpoint + generate + stream)
├── chat/           # Gradio UI (app.py), FastAPI (api.py), conversation, prompts
└── utils/          # logging_config.py, device.py, checkpoint.py
```

Key architectural decisions:
- **~157M params**: d_model=768, n_layers=14, n_heads=12, d_ff=3072
- **RoPE** positional encoding (LLaMA/Qwen/Mistral standard)
- **SwiGLU** activation (better than ReLU/GELU at same param count)
- **RMSNorm** normalization (faster than LayerNorm)
- **Pre-norm** residual connections (training stability)
- **Tied embeddings**: Token embedding and LM head share weights
- Custom **SentencePiece Unigram** tokenizer (32K vocab) trained on classical Chinese
- **FlashAttention** via `torch.nn.functional.scaled_dot_product_attention`

## Coding Standards

### Code Style & Quality
- **Function size**: Single functions must not exceed 50 lines. Complex logic must be decomposed into smaller, single-responsibility sub-functions.
- **File size**: Single Python files must not exceed 2,000 lines. Beyond this limit, split into modules or refactor into classes.
- **Naming conventions**:
  - Variables/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Type annotations**: All function signatures must include complete type annotations for parameters and return values. No omissions allowed.
- **Error handling**: Bare `except` blocks or `except: pass` are forbidden. Exceptions must be explicitly handled or logged.
- **Logging**: Use the `logging` module exclusively. `print()` is forbidden in production code.

### Project Structure
- `src/`: Source code
- `tests/`: Test code, mirroring the `src/` directory structure

### Development Workflow
- **Test-first**: Write or update unit tests before adding or modifying core logic.
- **Dependency management**: Use `pyproject.toml` for dependency management. Manual edits to `requirements.txt` are forbidden.
- **Code quality**: All commits must pass `black` formatting and `ruff` static analysis.

### Security
- **NEVER** hardcode passwords, API keys, or database connection strings in source code.
- **ALWAYS** set a `timeout` parameter when calling external APIs to prevent indefinite hangs.
- **NEVER** use `eval()` or `exec()` to execute dynamic code.
