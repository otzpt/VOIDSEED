# VOIDSEED

Training a language model from scratch on a single consumer GPU — to
understand how transformers actually work, not to compete with anything.

The model and training loop are written by hand. No `transformers`, no
`Trainer`, no prebuilt `GPT2LMHeadModel`. PyTorch provides autograd and CUDA
kernels; everything above that is written here.

## Status

| Stage | State |
|---|---|
| Environment (torch + CUDA verified) | done |
| `prepare.py` — data pipeline | done |
| `model.py` — transformer | not started |
| `train.py` — training loop | not started |
| `sample.py` — generation | not started |

**Corpus prepared:** TinyStories, 473,750,234 train tokens (904 MB uint16),
241,772 validation tokens.

## Sizing, from measured numbers

Training cost is roughly `6 × N × D` FLOPs (N = parameters, D = tokens), and
Chinchilla-optimal is ~20 tokens per parameter. The prepared corpus is
474M tokens, so:

```
474M tokens / 20 = ~24M parameters
```

**The first model targets ~24M.** Training something larger on this corpus
would be data-starved — a 100M model wants ~2B tokens, not 474M.

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
  capacity in the layers, less in a lookup table. A smaller custom vocab is a
  worthwhile second experiment.

### What the hardware allows

RTX 2080 SUPER, ~22 TFLOPS fp16, ~30% realistic utilisation → ~6.6e12
effective FLOP/s:

| Params | Tokens | FLOPs | Train time | Verdict |
|---|---|---|---|---|
| 24M | 10M | 1.4e15 | **~5 min** | Smoke test: does the loop run? |
| 24M | 100M | 1.4e16 | ~45 min | Undertrained but coherent |
| **24M** | **474M** | **6.8e16** | **~4 h** | This run — compute-optimal |
| 100M | 2B | 1.2e18 | ~50 h | A weekend, needs a bigger corpus |
| 1B | 20B | 1.2e20 | ~7 months | No |

Note that model size alone says little about training time — the corpus
dominates. A 24M model is small, but pushing 474M tokens through it is not
fast. Run the 10M-token smoke test first to shake out bugs; there is no
reason to discover a broken data loader four hours in.

Cost scales with N×D, so 10× the parameters costs ~100× the compute. That is
the whole reason frontier models cost eight figures and this one costs an
afternoon.

**Memory is not the constraint.** Mixed-precision Adam is ~16 bytes/param, so
even 100M params is ~1.6 GB in 8 GB of VRAM. Compute binds first.

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

python prepare.py --dataset tinystories
```

Verified with Python 3.14.4, torch 2.13.0+cu126.

## Layout

| File | |
|---|---|
| `prepare.py` | Streams a HuggingFace dataset, encodes with GPT-2 BPE, writes flat uint16 arrays split by document |
| `model.py` | Decoder-only transformer — causal self-attention, MLP, LayerNorm, residuals |
| `train.py` | memmap batching, cross-entropy next-token loss, AdamW, cosine LR with warmup |
| `sample.py` | Load a checkpoint and generate |

`prepare.py` is written because tokenization is plumbing. The rest is the
point of the project.

Data lives in `data/<dataset>/{train,val}.bin` as flat uint16 token ids, read
with `np.memmap` so the corpus never has to fit in RAM. It is `.gitignore`d —
it is *derived*, and regenerating it is one command.

## Roadmap

1. `model.py` at ~24M — `n_layer=6, n_head=6, n_embd=384, block_size=256`,
   tied embeddings
2. `train.py` — get the loss curve falling; that is the whole feedback loop.
   Smoke-test on ~10M tokens (~5 min) before committing to the full run
3. `sample.py` — generate; expect grammatical, mostly coherent short stories
4. Then experiment: smaller vocabulary, different depth/width at fixed
   parameter count, learning-rate schedules

## What to expect

A 24M model trained on TinyStories writes simple, mostly coherent children's
stories. It **cannot** code, follow instructions, or use tools — those need
orders of magnitude more compute plus instruction tuning. This is built to
understand transformers, not to use one.

## License

MIT. Dependencies are permissive throughout: PyTorch (BSD-3), tiktoken (MIT),
datasets (Apache-2.0), numpy (BSD). TinyStories is MIT; FineWeb-Edu is ODC-By.
