# Classic Chinese LLM

A classical Chinese (文言文) conversational LLM built from scratch to learn Transformers. A ~157M parameter GPT-2-style Decoder-only Transformer is implemented using only `torch.nn`, then pretrained and instruction-tuned on classical Chinese corpora.

## Requirements

- Python 3.12
- NVIDIA GPU with 12GB+ VRAM, BF16 support

## Quick Start

```bash
# Create a Python 3.12 environment, then:
pip install -e ".[data,chat,dev]"

# Collect and preprocess data
python scripts/collect_data.py --output-dir data/raw
python scripts/train_tokenizer.py --corpus data/processed/cleaned.txt --vocab-size 32000

# Train
python scripts/pretrain.py --config configs/pretrain.yaml    # ~2-3 days
python scripts/finetune.py --config configs/sft.yaml          # ~2-4 hours

# Chat
python scripts/chat.py --checkpoint models/checkpoints/sft_best.pt
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full architecture design.
