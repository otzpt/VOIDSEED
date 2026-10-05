# VOIDSEED 

VOIDSEED is a small GPT style LLM coded and trained from scratch in python using
pytorch.

No transformers library nor trainer was used, the attention, training loop
and retrieval layer were written by me

I decided to do it this way so i can learn how these things actually work
instead of just knowing how to do it, also because this way i have more control
over what the actual code does instead of just accepting whatever is thrown at me

## Known limitation

So this is important, considering the size of the model and that it is under trained for its size (1.87B tokens for 155M parameters) the model is incapable of maintaining a normal conversation nor give fully correct answers, as it was tested in around 10 cybersecurity prompts the model got all the 10 wrong or partially correct, don't rely on this model for anything, if the model give an answer like "``______``" don't worry it's not broken nor it's a bug, the model wasn't trained on any conversation docs or anything like that so it just uses place holder or separation tokens which are very common in its training docs 

## Phases

So this LLM went through 2 phases a smaller test phase and the "serious" one.

 - Phase 1: A smaller 24M parameters model trained using tinyStories made just to check if the training loop actually worked or not before spending 20+ hours training the model (this alone took 6 hours to train)
 - Phase 2: The actual LLM, 155M parameters, though it's small its still very impressive (for all the people that want to know: `d_model=768, n_heads=12, n_layer=12, block_size=256`, from `train.py`), this was trained on around 2B tokens of cybersecurity docs

## What security docs?

The list is in `repos/`: hacktricks, PayloadsAllTheThings, SecLists,
GTFOBins, CheatSheetSeries, wstg.

`prepare.py` trains only on files like md/rst/yml not raw wordlists, the main reason is that a wordlist file would be thousands of lines with no real sentences just raw info which would just drown everything else 100:1

### Quick Notice

`prepare.py` was outsourced to claude code

## The model itself

This, at least for me, is the fun part.

The model itself is a decoder-only transformer, GPT-2 small sized with 12 stacked blocks, text becomes token IDs, then vectors plus a position vector so it know the word order. Each token looks at earlier tokens to decide what matters, the caulsal mask stop it from seeing what it shouldn't(like the future for example), it runs 12 heads in parallel then merges them, after attention a small MLP layer processes each token, Residual connections and LayerNorm keep training stable, now the final layer, it scores all 50257 tokens, and the next one is picked form those scores.

### Extra

I added KV_cache with the help of claude so generation reuses past work also to speep up thinking and thing like that, after kv_cache answer time went from 54~ seconds to aroiun 4 to 12 sec

## Training

The training was made using mixed precision(fp16) with gradient scaling, learning-rate schedule with warmup, checkpoints every 1000 steps to make sure that in case of an outage i didnt lose training progress and it only keep last 5 so i dont run out of storage(it happened one time) and in case the training gets interrupted or stopped for any reason it resumes from latest checkpoint if there is one available

## Retrieval

`search_index.py` builds a full-text search using SQlite FTS5, when you chat with model the query is matched against the index, the thing that matches best gets pulled as context after that the model generates the answer using that context.

I also did some filtering such as dropping link-heavy chunks, stripping URLs from the model output so it cant just invent fake-looking links.

I also did sampling tuning like top-k + repetition penalty + temperature and i actually measured which settings give more accurate answers instead of eye balling it

## Hardware

This is also a fun part, so the model was trained on my main pc using a RTX 2080 Super with fp16 instead of bf16 why you might ask, well because Turing GPUs dont have real bf16 tensor cores, so bf16 gets emulated which is slower than fp16 because it emulated

## How to run it

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
