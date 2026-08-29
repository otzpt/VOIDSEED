#!/usr/bin/env python3
"""Build a corpus and tokenize it into flat uint16 arrays.

This is the boring half of the project — deliberately written for you so the
interesting half (the model and the training loop) is yours to write.

Output is two memory-mappable files:

    data/<dataset>/train.bin
    data/<dataset>/val.bin

Each is a flat array of uint16 GPT-2 BPE token ids. uint16 works because the
GPT-2 vocabulary is 50257 tokens, which fits under 65536. Training reads these
with np.memmap, so the corpus never has to fit in RAM.

Two kinds of source:

  tinystories — streamed from the Hub, unchanged from the original version
  of this file.

  security — built from whatever repos are cloned under repos/. This script
  only needs to know WHERE to look, not WHAT is there: any directory dropped
  under repos/ is picked up automatically next run, nothing here needs
  editing to add another source.

Usage:
    python prepare.py --dataset tinystories          # ~470M tokens, start here
    python prepare.py --dataset security             # everything under repos/
"""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent

DATASETS = {
    # Small, clean, and the model learns coherent English fast. The right
    # first target: you can train something that works in under an hour.
    "tinystories": dict(path="roneneldan/TinyStories", name=None, text_key="text"),
}

# Prose and structured technique data: markdown wikis, GTFOBins' per-binary
# YAML entries, that kind of thing. This is the default allowlist for the
# `security` dataset.
LOCAL_TEXT_EXTENSIONS = {".md", ".rst", ".adoc", ".yml", ".yaml"}

# .txt is gated behind --include-txt rather than included by default. It is
# genuinely ambiguous in this domain: PayloadsAllTheThings' .txt files are
# short fuzzing payload fragments, but SecLists' are one-word-per-line
# wordlists — 6,042 files, 1.7 GB, versus ~17 MB of everything else combined.
# Included by default, SecLists alone would outweigh every other source
# roughly 100:1, and the corpus would teach "the statistical shape of a
# password list" instead of "explains a technique", which is a different
# model for a different job.
LOCAL_RAW_EXTENSIONS = {".txt"}

# mdbook's include directive, which hacktricks uses for a shared promo banner
# on nearly every page (991 of 998 files in the repo as cloned 2026-08).
# Templating syntax, not content — left in, it becomes the single most
# repeated string in the corpus for no benefit.
_MDBOOK_INCLUDE = re.compile(r"\{\{#include[^}]*\}\}")


def clean_local_text(text: str) -> str:
    return _MDBOOK_INCLUDE.sub("", text)


def iter_repo_texts(repos_dir: pathlib.Path, include_txt: bool):
    """Yield (repo_name, text) for every matching file under repos_dir/*/.

    Walks whatever is actually there rather than a fixed list of repo names,
    so `git clone` into repos/ is the only step needed to add a source.
    """
    if not repos_dir.is_dir():
        return
    extensions = LOCAL_TEXT_EXTENSIONS | (LOCAL_RAW_EXTENSIONS if include_txt else set())
    for path in sorted(repos_dir.rglob("*")):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        repo_name = path.relative_to(repos_dir).parts[0]
        yield repo_name, clean_local_text(text)


def _tokenize_stream(texts, enc, eot, f_train, f_val, rng, val_frac, limit,
                     bar, n_train, n_val):
    """Tokenize an iterable of document strings, splitting by document (not
    by token, so one document never straddles the train/val boundary and
    leaks) into the given open files.

    Shared by both corpus sources — streamed and local — so this split/write
    logic exists exactly once. Returns the updated (n_train, n_val) and
    whether `limit` was reached.
    """
    for text in texts:
        if not text:
            continue
        ids = enc.encode_ordinary(text)
        ids.append(eot)
        arr = np.array(ids, dtype=np.uint16)

        if rng.random() < val_frac:
            arr.tofile(f_val)
            n_val += arr.size
        else:
            arr.tofile(f_train)
            n_train += arr.size

        bar.update(arr.size)
        if limit and (n_train + n_val) >= limit:
            return n_train, n_val, True
    return n_train, n_val, False


def prepare_security(args, enc, eot, out_dir) -> tuple[int, int]:
    if args.include_txt:
        print("--include-txt set: wordlists and raw payload lists will be "
              "ingested as prose. This changes what the model learns from "
              "'explains a technique' toward 'the statistical shape of a "
              "wordlist' wherever those files dominate a repo (SecLists in "
              "particular). See the comment on LOCAL_RAW_EXTENSIONS.")

    train_path, val_path = out_dir / "train.bin", out_dir / "val.bin"
    rng = np.random.default_rng(1337)
    n_train = n_val = 0
    per_repo: dict[str, int] = {}

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        bar = tqdm(unit="tok", unit_scale=True, desc="tokenizing repos/")

        # Grouped by repo, rather than tokenized as one flat stream, purely so
        # the per-repo token report below is exact.
        by_repo: dict[str, list[str]] = {}
        for repo_name, text in iter_repo_texts(args.repos_dir, args.include_txt):
            by_repo.setdefault(repo_name, []).append(text)

        if not by_repo:
            print(f"\nno files found under {args.repos_dir} "
                  f"(extensions: {sorted(LOCAL_TEXT_EXTENSIONS)}"
                  f"{' + .txt' if args.include_txt else ''}). "
                  f"Nothing to clone in yet? repos/<name>/ is picked up "
                  f"automatically once something is.")
            bar.close()
            return 0, 0

        hit_limit = False
        for repo_name, texts_in_repo in by_repo.items():
            if hit_limit:
                break
            before = n_train + n_val
            n_train, n_val, hit_limit = _tokenize_stream(
                texts_in_repo, enc, eot, f_train, f_val, rng,
                args.val_frac, args.limit, bar, n_train, n_val)
            per_repo[repo_name] = (n_train + n_val) - before

        bar.close()

    print("\nper-repo token counts:")
    for repo_name, count in sorted(per_repo.items(), key=lambda kv: -kv[1]):
        print(f"  {repo_name:<30} {count:>14,}")

    return n_train, n_val


def prepare_hf(args, enc, eot, out_dir, spec, f_train, f_val, rng, bar,
               n_train, n_val) -> tuple[int, int, bool]:
    print(f"streaming {spec['path']} ...")
    ds = load_dataset(spec["path"], spec["name"], split="train", streaming=True)
    texts = (row.get(spec["text_key"]) for row in ds)
    return _tokenize_stream(texts, enc, eot, f_train, f_val, rng,
                            args.val_frac, args.limit, bar, n_train, n_val)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS) + ["security"],
                    default="tinystories")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after roughly this many tokens")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="fraction held out for validation (default: 0.0005 "
                         "for streamed HF datasets, 0.02 for security — see "
                         "the comment where this is resolved)")
    ap.add_argument("--repos-dir", type=pathlib.Path, default=ROOT / "repos",
                    help="[security] where cloned repos live (default: ./repos)")
    ap.add_argument("--include-txt", action="store_true",
                    help="[security] also ingest .txt files. Off by default "
                         "— see the comment on LOCAL_RAW_EXTENSIONS in this "
                         "file for why.")
    args = ap.parse_args()

    if args.val_frac is None:
        # 0.0005 assumes millions of documents, true for the HF streaming
        # datasets. security has on the order of 1,700 files: at 0.0005, the
        # expected val set is under one file, and measured runs during
        # development landed on val.bin being empty outright. 0.02 (~2%,
        # ~34 files) is small enough not to starve training and reliably
        # non-empty.
        args.val_frac = 0.02 if args.dataset == "security" else 0.0005

    out_dir = ROOT / "data" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # marks document boundaries so docs don't bleed together

    if args.dataset == "security":
        n_train, n_val = prepare_security(args, enc, eot, out_dir)
    else:
        spec = DATASETS[args.dataset]
        train_path, val_path = out_dir / "train.bin", out_dir / "val.bin"
        rng = np.random.default_rng(1337)
        with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
            bar = tqdm(unit="tok", unit_scale=True, desc="tokenizing")
            n_train, n_val, _ = prepare_hf(
                args, enc, eot, out_dir, spec, f_train, f_val, rng, bar, 0, 0)
            bar.close()

    train_path, val_path = out_dir / "train.bin", out_dir / "val.bin"
    print(f"\ntrain : {n_train:,} tokens  ->  {train_path}")
    print(f"val   : {n_val:,} tokens  ->  {val_path}")
    print(f"size  : {(n_train + n_val) * 2 / 1e9:.2f} GB on disk (uint16)")

    if args.limit and (n_train + n_val) < args.limit:
        print(f"\nshort of the {args.limit:,}-token target by "
              f"{args.limit - (n_train + n_val):,} tokens. For `security`, "
              f"that means repos/ did not have enough content — clone more "
              f"sources into repos/.")

    # Chinchilla says ~20 tokens per parameter is compute-optimal.
    optimal = (n_train + n_val) / 20
    print(f"\nAt ~20 tokens/param, this corpus is compute-optimal for a "
          f"~{optimal/1e6:.0f}M parameter model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
