# VOIDSEED

A language model from scratch in ``python`` and ``pytorch``, no
``transformers`` lib, no ``Trainer``, wrote the model, the training loop and
the retrieval part myself.

There were two phases. First one was just a small 24M model on TinyStories
to check if the training loop actually worked or if i had to fix anything due to skillissue. 
Second one is the real thing: a 155M model trained on ~2B tokens of security docs, plus a retrieval layer so
it can actually quote real sources instead of making things up.

## Try it

Weights are on **[v1.0-weights release](https://github.com/otzpt/VOIDSEED/releases/tag/v1.0-weights)**
— `model_976000.pt`, 622 MB, no optimizer state, just the weights.

## Phase 2 — the actual model

**Corpus:** everything under `repos/` — hacktricks, PayloadsAllTheThings,
SecLists, GTFOBins, CheatSheetSeries, wstg. Only markdown/rst/yaml files by
default, not the raw wordlists, or SecLists alone would drown out
everything else.

Measured from the actual run: 1,960,108,296 train tokens, 39,891,739 val
tokens, 4GB on disk.

Model is `d_model=768, n_heads=12, n_layer=12, block_size=256`, GPT-2-small
sized. 155,410,513 params, counted from the checkpoint.

```bash
python prepare.py --dataset security
python prepare.py --dataset security --include-txt  # not recommended, see above
```

## Retrieval

`chat.py` doesn't just generate tokens, it searches an FTS5 SQLite index over the
same corpus, grabs the top matches, gets them in as context, then
generates. `search_index.py` builds the index.

```bash
python search_index.py
python chat.py
```

## Phase 1 — the sizing test

rule: cost is `6 x params x tokens` FLOPs, and somewhere around 20 tokens per parameter is
supposed to be compute-optimal. 474M tokens / 20 = ~24M params, so that's
what I trained first.

## Hardware

RTX 2080 SUPER. Use fp16, not bf16, Turing doesn't have real bf16 tensor
cores so it just emulates it so because of that it's slower.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy tiktoken datasets tqdm matplotlib

python prepare.py --dataset tinystories
python prepare.py --dataset security
```
