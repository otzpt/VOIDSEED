# VOIDSEED

Training a language model from scratch, to understand the machine rather than
to compete with it.

## Why the scale is what it is

Training cost is roughly `6 × N × D` FLOPs (N = parameters, D = tokens), and
Chinchilla-optimal is ~20 tokens per parameter. On an RTX 2080 SUPER
(~22 TFLOPS fp16, ~30% realistic utilisation → ~6.6e12 effective FLOP/s):

| Params | Tokens | FLOPs | Train time | Verdict |
|---|---|---|---|---|
| 10M | 200M | 1.2e16 | **~30 min** | Start here |
| 100M | 2B | 1.2e18 | **~50 h** | A weekend |
| 350M | 7B | 4.2e18 | ~3 weeks | Painful |
| 1B | 20B | 1.2e20 | ~7 months | No |

That table is the lesson. Cost scales with N×D, so 10× the parameters costs
~100× the compute. It's why frontier models cost eight figures.

**Memory is not the constraint.** Mixed-precision Adam is ~16 bytes/param, so
100M params ≈ 1.6 GB — comfortable in 8 GB. Compute binds long before VRAM.

## Hardware notes (RTX 2080 SUPER, compute capability 7.5)

- **Use fp16 with `torch.amp.GradScaler`.** `torch.cuda.is_bf16_supported()`
  returns `True`, but Turing has no native bf16 tensor cores — it's emulated
  and slow. Tutorials that default to bf16 will quietly halve your throughput.
- 8 GB VRAM. Batch size × sequence length is what you'll actually tune.

## Layout

| File | Status |
|---|---|
| `prepare.py` | **Provided.** Downloads and tokenizes into `data/<set>/{train,val}.bin` |
| `model.py` | **Yours.** Embeddings, attention, MLP, blocks |
| `train.py` | **Yours.** Batching, loss, optimizer, checkpoints |
| `sample.py` | **Yours.** Load a checkpoint and generate |

`prepare.py` is written for you because tokenization is plumbing, not
learning. The model and the training loop are the point — write those yourself.

## Order of work

1. **`prepare.py --dataset tinystories`** — confirm the data pipeline before
   writing any model code.
2. **`model.py`** — a decoder-only transformer. Causal self-attention, MLP,
   LayerNorm, residuals, learned positional embeddings. Start with ~10M params
   (6 layers, 6 heads, 384 dim).
3. **`train.py`** — memmap the `.bin`, sample random windows, cross-entropy on
   next-token prediction, AdamW, cosine LR schedule with warmup. **Print the
   loss.** Watching it fall is the whole feedback loop.
4. **`sample.py`** — generate text. At 10M on TinyStories you should get
   grammatical, mostly coherent short stories.
5. Only then scale to 100M.

## What to expect

A 100M model writes fluent English and coherent short passages. It **cannot**
code, follow instructions, or use tools — those need 1000× the compute and
instruction tuning on top. Build it to understand transformers, not to use it.

## Storage

Data and checkpoints stay local (`.gitignore`d). Tokenized corpora run to
gigabytes and GitHub rejects files over 100 MB. The data is *derived* — if you
lose it, re-run `prepare.py`. What's worth committing is the code.
