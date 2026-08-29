# VOIDSEED

Training a language model from scratch — to understand how transformers
actually work, not to compete with anything.

The model, training loop, and retrieval layer are written by hand. No
`transformers`, no `Trainer`, no prebuilt `GPT2LMHeadModel`. PyTorch provides
autograd and CUDA kernels; everything above that is written here.

Two phases happened here, in order. Phase 1 (below) is the sizing exercise
and the smoke test. Phase 2 is the actual result: a 155M-parameter model
trained on ~2B tokens of security documentation, paired with a full-text
retrieval index so answers are grounded in real reference material instead
of whatever the model happened to memorize.

## Try it

Weights: **[v1.0-weights release](https://github.com/otzpt/VOIDSEED/releases/tag/v1.0-weights)**
— `model_976000.pt`, 622 MB, weights only (no optimizer state). See the
release notes for the architecture config and exact loading code.

## Status

| Stage | State |
|---|---|
| `prepare.py` — data pipeline (`datasets` library + local `security` corpus) | done |
| `model.py` — transformer | done |
| `train.py` — training loop (fp16, grad clipping, cosine LR, checkpointing) | done |
| **Phase 1** — 24M params, TinyStories, smoke test | done — reached step 231,000 |
| **Phase 2** — 155M params, security corpus, full run | done — step 976,000 / 976,562 |
| `chat.py` — interactive generation with retrieval | done |
| `search_index.py` — FTS5 full-text index over the training repos | done |

## Phase 2 — the security model (current)

**Corpus:** everything under `repos/` — real, cloned reference repos, not
synthetic data: `hacktricks`, `hacktricks-cloud`, `PayloadsAllTheThings`,
`SecLists`, `GTFOBins.github.io`, `CheatSheetSeries`, `wstg`. Prose and
structured technique files only by default (`.md`, `.rst`, `.adoc`, `.yml`,
`.yaml`) — raw wordlists are gated behind `--include-txt` because SecLists
alone would otherwise outweigh every other source ~100:1 and the model would
learn "the statistical shape of a password list" instead of "explains a
technique." See the comment on `LOCAL_RAW_EXTENSIONS` in `prepare.py`.

**Measured, from `logs/prepare-blend.log`:**

| | |
|---|---|
| Train tokens | 1,960,108,296 |
| Val tokens | 39,891,739 |
| On disk (uint16) | 4.00 GB |

**Model:** `d_model=768, n_heads=12, n_layer=12, block_size=256` — GPT-2-small
sized. 155,410,513 parameters (measured from the checkpoint's state dict).

At ~20 tokens/param (Chinchilla-optimal), this corpus is compute-optimal for
a ~100M-parameter model. 155M against 1.96B tokens is ~12.6 tokens/param —
somewhat below optimal, so the model is a bit undertrained relative to its
size rather than the corpus being too small for it. Worth revisiting: either
a smaller model on this same corpus, or more corpus for this model size.

**Training schedule:** cosine LR, peak `3e-4`, 2,000-step warmup,
`max_steps = 976,562`. The checkpoint in `checkpoints/` (`976000.pt`) is at
step 976,000 — 99.94% through the schedule.

```bash
python prepare.py --dataset security                 # everything under repos/
python prepare.py --dataset security --include-txt    # + raw wordlists (not recommended, see above)
```

## Retrieval (`search_index.py`, `chat.py`)

`chat.py` doesn't just sample from the model — it searches an FTS5 SQLite
index built from the same `repos/` corpus, injects the top-k matching
passages as context ahead of the question, then generates. `search_index.py`
is deliberately broader than the training allowlist: it indexes `.txt` too,
since a wordlist is useless training signal but genuinely useful to look up
("common admin passwords") when a query actually matches it. See the module
docstring in `search_index.py` for the reasoning.

```bash
python search_index.py                              # (re)build search.db from repos/
python search_index.py --query "sql injection" -k 3  # try a lookup directly
python chat.py                                       # interactive REPL, model + retrieval
```

## Phase 1 — sizing exercise, TinyStories (24M)

Training cost is roughly `6 × N × D` FLOPs (N = parameters, D = tokens), and
Chinchilla-optimal is ~20 tokens per parameter.

```
474M tokens / 20 = ~24M parameters
```

The first model targeted ~24M params against the prepared TinyStories corpus
(473,750,234 train tokens, 904 MB uint16). Training something larger on that
corpus would have been data-starved — a 100M model wants ~2B tokens, which is
exactly the scale phase 2 moved to on a different corpus.

### Where those parameters actually go

This is the part that surprises people. With GPT-2's 50,257-token vocabulary
at `n_embd = 384`:

| Component | Parameters |
|---|---|
| Token embeddings (50257 × 384) | **19.3M** |
| Position embeddings (256 × 384) | 0.1M |
| 6 transformer blocks (12 × L × d²) | 10.6M |

The embedding table is larger than the entire transformer. Two conclusions:

- **Tie the embedding and output-head weights.** Standard practice, and it
  avoids spending another 19.3M on the unembedding.
- GPT-2's vocabulary was built for all of internet English. TinyStories is
  deliberately simple prose, so most of those 50k tokens never appear. The
  original TinyStories paper used a ~10k vocabulary for this reason — more
  capacity in the layers, less in a lookup table.

### What the hardware allows

RTX 2080 SUPER, ~22 TFLOPS fp16, ~30% realistic utilisation → ~6.6e12
effective FLOP/s:

| Params | Tokens | FLOPs | Train time | Verdict |
|---|---|---|---|---|
| 24M | 10M | 1.4e15 | **~5 min** | Smoke test: does the loop run? |
| 24M | 100M | 1.4e16 | ~45 min | Undertrained but coherent |
| **24M** | **474M** | **6.8e16** | **~4 h** | Phase 1 — compute-optimal for TinyStories |
| 100M | 2B | 1.2e18 | ~50 h | A weekend |
| 155M | 1.96B | ~1.4e18 | ~55–60 h (est., same GPU) | Phase 2 — no wall-clock log kept, this is inferred from the FLOPs table above, not measured |
| 1B | 20B | 1.2e20 | ~7 months | No |

Cost scales with N×D, so 10× the parameters costs ~100× the compute. That is
the whole reason frontier models cost eight figures and this one costs an
afternoon (phase 1) or a couple of days (phase 2).

**Memory is not the constraint.** Mixed-precision Adam is ~16 bytes/param, so
even 155M params is ~2.5 GB in 8 GB of VRAM. Compute binds first.

## Hardware notes (Turing, compute capability 7.5)

Use **fp16 with `torch.amp.GradScaler`**. `torch.cuda.is_bf16_supported()`
returns `True` on this card, but Turing has no native bf16 tensor cores — it
is emulated and slow. Tutorials that default to bf16 will quietly halve
throughput.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy tiktoken datasets tqdm matplotlib

python prepare.py --dataset tinystories     # phase 1 corpus
# or, for phase 2:
# git clone each repo listed under "Corpus" above into repos/<name>
python prepare.py --dataset security
```

Verified with Python 3.14.4, torch 2.13.0+cu126.

## Layout

| File | |
|---|---|
| `prepare.py` | Streams a dataset via the `datasets` library, or walks `repos/` for the `security` corpus; encodes with GPT-2 BPE, writes flat uint16 arrays split by document |
| `model.py` | Decoder-only transformer — causal self-attention, MLP, LayerNorm, residuals |
| `train.py` | memmap batching, cross-entropy next-token loss, AdamW, cosine LR with warmup, fp16 |
| `chat.py` | Interactive REPL: retrieves context via `search_index.py`, then generates |
| `search_index.py` | FTS5 SQLite full-text index over `repos/` — the retrieval half of `chat.py` |
| `strip_checkpoint.py` (in `../voidseed-tinyllm-space/`) | Drops optimizer state from a training checkpoint for deployment — 1.86GB → 622MB, measured |

`prepare.py` is written because tokenization is plumbing. The rest is the
point of the project.

Data lives in `data/<dataset>/{train,val}.bin` as flat uint16 token ids, read
with `np.memmap` so the corpus never has to fit in RAM. It is `.gitignore`d —
it is *derived*, and regenerating it is one command (`repos/` for phase 2 is
likewise not committed — clone the repos listed under "Corpus" above).

## What to expect

Phase 1 (24M, TinyStories) writes simple, mostly coherent children's stories.
Phase 2 (155M, security corpus + retrieval) answers security/pentesting
questions grounded in the retrieved passages — coherent on a good day, still
a 155M model, not a substitute for reading the actual source material it's
quoting from. Neither phase can follow complex multi-step instructions or use
tools — that needs orders of magnitude more compute plus instruction tuning.
This is built to understand transformers and retrieval, not to replace either
hacktricks or a frontier model.

## License

MIT. Dependencies are permissive throughout: PyTorch (BSD-3), tiktoken (MIT),
datasets (Apache-2.0), numpy (BSD). TinyStories is MIT. The `repos/` sources
(hacktricks, PayloadsAllTheThings, SecLists, GTFOBins, CheatSheetSeries, wstg)
each carry their own license — check before
redistributing the trained weights or the search index built from them.
