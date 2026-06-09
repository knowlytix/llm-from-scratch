# Building Language Models from Scratch

*From the chain rule to a deployable model — no black boxes.*

Instead of calling a library and treating the model as magic, you build every
piece yourself: byte-level BPE, causal self-attention, the Transformer block, a
training loop with warmup-cosine and diagnostics, a KV-cache inference engine, an
evaluation harness, and a fine-tuning stack (SFT, LoRA, DPO). Each chapter is
paired with a notebook that runs end-to-end on a single GPU and **produces every
figure the chapter cites** — you see the real outputs, not idealized diagrams. The
library is small (~5,000 LOC), inspectable, and backed by **190 tests**; the GRU
is verified against PyTorch's reference, and DPO comes with a proof.

> Part of the **"Beyond … and Pray"** series:
> [governed agents](https://github.com/knowlytix/beyond-prompt-and-pray) ·
> [trustworthy RAG](https://github.com/knowlytix/beyond-chunk-and-pray) ·
> [test & validate](https://github.com/knowlytix/beyond-ship-and-pray) ·
> [LLMs from scratch](https://github.com/knowlytix/llm-from-scratch)

## What's inside

- **Tokenization** — character / word / byte + byte-level BPE from scratch
- **Transformers** — attention, multi-head, RoPE / ALiBi / sinusoidal / learned, TinyGPT
- **Training** — `Trainer`, scaling laws, mixed precision, diagnostics
- **Inference** — greedy / beam / top-k / top-p + a **KV-cache engine**
- **Evaluation** — perplexity, calibration, LLM-as-judge, contamination checks
- **Fine-tuning** — SFT (loss-masked), **LoRA from scratch**, **DPO with a proof**
- **190 tests**, **33 notebooks**, one per chapter

## Install

```bash
pip install llm-from-scratch              # core: torch, numpy, scikit-learn
pip install "llm-from-scratch[notebooks]" # + jupyterlab, matplotlib, pandas
pip install "llm-from-scratch[all]"       # everything
```

Python 3.12+. A CUDA-capable GPU is recommended for the training chapters.

## Quickstart

Every piece is built from scratch and importable on its own — e.g. the byte-level
/ char tokenizer:

```python
from llm_from_scratch.tokenizers.char_tokenizer import CharTokenizer

tok = CharTokenizer()
tok.train("hello world")
ids = tok.encode("hello")
print(ids, "->", tok.decode(ids), "| vocab:", tok.vocab_size)
# [7, 6, 8, 8, 9] -> hello | vocab: 12
```

From here the chapters build up the Transformer, training loop, KV-cache
inference, evaluation, and fine-tuning — each paired with a runnable notebook in
[`notebooks/`](notebooks/).

## License

Apache-2.0. © 2026 Knowlytix.
