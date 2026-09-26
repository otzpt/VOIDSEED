# README rewrite — what to cover

Reviewer rejected the old README because it reads AI-written. Write every
section below yourself, in your own words. This file is a checklist of
what to say, based on reading your actual code just now, not what to
copy-paste.

## 1. What this project is
- A GPT-style language model, written from scratch in PyTorch.
- No `transformers` library, no `Trainer`. You wrote the attention, the
  training loop, and the retrieval layer yourself.
- Say why you did it this way instead of using a library (learning, or
  wanting full control — whichever is true for you).

## 2. The two phases, and why
- Phase 1: a small 24M-param model trained on TinyStories, just to check
  your training loop actually worked before spending real compute.
- Phase 2: the real model. 155M params, GPT-2-small sized
  (`d_model=768, n_heads=12, n_layer=12, block_size=256`, from `train.py`).
  Trained on ~2 billion tokens of security docs.
- Explain in your own words why you picked this order (cheap sanity check
  before the expensive run).

## 3. What "security docs" means
- The corpus is `repos/`: hacktricks, PayloadsAllTheThings, SecLists,
  GTFOBins, CheatSheetSeries, wstg.
- `prepare.py` only trains on prose files (md/rst/yaml), not raw
  wordlists — explain why (a wordlist file is thousands of lines with no
  real sentences, it would drown out everything else 100:1).

## 4. The model itself (model.py)
Explain each piece in plain words, not the code:
- Causal self-attention: each token can only look at tokens before it,
  not after (that's what "causal" means here, and why there's a mask).
- Multi-head attention: splits the vector into several smaller heads that
  attend independently, then merges them back.
- KV cache: during generation, past keys/values are reused instead of
  recomputed every step. Say why that matters (speed, one new token at a
  time instead of reprocessing the whole sequence).
- Residual connections (`x + attn_out`): why you keep the "x +" — skipping
  it lets one bad layer wreck everything after it.
- You even wrote a self-check at the bottom of `model.py` that verifies the
  cached path gives the same output as the full-sequence path — mention
  this, it's a real engineering habit worth showing off.

## 5. Training (train.py)
- Mixed precision (fp16) training with gradient scaling.
- Cosine learning-rate schedule with warmup.
- Checkpointing every 1000 steps, keeps only the last 5 (say why — disk
  space, a 150M model checkpoint isn't small).
- Resumes from the latest checkpoint automatically if one exists.

## 6. Retrieval (search_index.py + chat.py)
- `search_index.py` builds a full-text search index (SQLite FTS5) over the
  same corpus.
- At chat time, your query is matched against that index, the best-matching
  passages get pulled in as context, and the model generates its answer
  grounded in that context — so it's not just making things up.
- Mention the filtering you did: dropping link-heavy chunks (tables of
  contents), stripping URLs from the model's own output so it can't
  invent fake-looking links.
- Mention the sampling tuning: top-k + repetition penalty + temperature,
  and that you actually measured which settings kept answers grounded in
  the retrieved text (this is a real, specific result — say the numbers).

## 7. Hardware
- RTX 2080 SUPER. fp16, not bf16 — Turing GPUs don't have real bf16
  tensor cores, so bf16 gets emulated and runs slower. Say this in your
  own words.

## 8. How to run it
- Setup, `prepare.py` for each dataset, `search_index.py` to build the
  index, `chat.py` to talk to it. Keep this short, it's just commands.

---

Write the actual README in your own voice. Short and rough is fine —
that's the point, it has to sound like you.

---

# DRAFT: How to run it (reword in your own voice)

PROBLEM TO FIX FIRST: the release file `model_976000.pt` is a bare
state_dict (the release notes say "state_dict only, no optimizer state, no
step counter"). `chat.py` line 27-30 does `checkpoint['model']`, which only
exists in the full 1.78 GB training checkpoint. Following the steps below
with the release file will fail in `chat.py`. Not run, but that is what the
code reads. Fix `chat.py` (or the release) before you write step 5 down.

## Steps

1. Clone the repo:
   `git clone https://github.com/otzpt/VOIDSEED.git`, then `cd VOIDSEED`.
2. Environment (Linux): `bash setup.sh`, then `source .venv/bin/activate`.
   No NVIDIA GPU: `TORCH_INDEX=cpu bash setup.sh`.
   Windows: `python -m venv .venv`, `.venv\Scripts\activate`,
   `pip install torch --index-url https://download.pytorch.org/whl/cu126`,
   `pip install -r requirements.txt`.
3. Clone the corpus into `repos/` (git-ignored, so it is not in the repo).
   Sources: OWASP/CheatSheetSeries, GTFOBins/GTFOBins.github.io,
   HackTricks-wiki/hacktricks, HackTricks-wiki/hacktricks-cloud,
   swisskyrepo/PayloadsAllTheThings, danielmiessler/SecLists, OWASP/wstg
   (all under github.com). SecLists is large.
4. Build the search index: `python search_index.py`. Writes `search.db`
   (git-ignored, about 3.6 GB on my machine).
5. Get the weights: download `model_976000.pt` from the v1.0-weights release
   and save it as `checkpoints/976000.pt` (path is hardcoded in `chat.py`).
6. Chat: `python chat.py`. Type a question, `exit` to quit.

Only if retraining: `python prepare.py --dataset security`, then
`python train.py`. Not needed to use the model.
