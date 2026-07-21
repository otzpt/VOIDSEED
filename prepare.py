#!/usr/bin/env python3
"""Download a corpus and tokenize it into flat uint16 arrays.

This is the boring half of the project — deliberately written for you so the
interesting half (the model and the training loop) is yours to write.

Output is two memory-mappable files:

    data/<dataset>/train.bin
    data/<dataset>/val.bin

Each is a flat array of uint16 GPT-2 BPE token ids. uint16 works because the
GPT-2 vocabulary is 50257 tokens, which fits under 65536. Training reads these
with np.memmap, so the corpus never has to fit in RAM.

Usage:
    python prepare.py --dataset tinystories     # ~470M tokens, start here
    python prepare.py --dataset fineweb-edu --limit 2_000_000_000
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent

DATASETS = {
    # Small, clean, and the model learns coherent English fast. The right
    # first target: you can train something that works in under an hour.
    "tinystories": dict(path="roneneldan/TinyStories", name=None, text_key="text"),
    # Real web text, heavily filtered for educational content. Use when you
    # want a 100M-param run that produces something genuinely useful.
    "fineweb-edu": dict(
        path="HuggingFaceFW/fineweb-edu", name="sample-10BT", text_key="text"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="tinystories")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after roughly this many tokens")
    ap.add_argument("--val-frac", type=float, default=0.0005,
                    help="fraction held out for validation")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    out_dir = ROOT / "data" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # marks document boundaries so docs don't bleed together

    print(f"streaming {spec['path']} ...")
    ds = load_dataset(spec["path"], spec["name"], split="train", streaming=True)

    train_path, val_path = out_dir / "train.bin", out_dir / "val.bin"
    n_train = n_val = 0
    rng = np.random.default_rng(1337)

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        bar = tqdm(unit="tok", unit_scale=True, desc="tokenizing")
        for row in ds:
            text = row.get(spec["text_key"])
            if not text:
                continue
            ids = enc.encode_ordinary(text)
            ids.append(eot)
            arr = np.array(ids, dtype=np.uint16)

            # Split by document, not by token, so a story never straddles the
            # train/val boundary and leaks.
            if rng.random() < args.val_frac:
                arr.tofile(f_val)
                n_val += arr.size
            else:
                arr.tofile(f_train)
                n_train += arr.size

            bar.update(arr.size)
            if args.limit and (n_train + n_val) >= args.limit:
                break
        bar.close()

    print(f"\ntrain : {n_train:,} tokens  ->  {train_path}")
    print(f"val   : {n_val:,} tokens  ->  {val_path}")
    print(f"size  : {(n_train + n_val) * 2 / 1e9:.2f} GB on disk (uint16)")

    # Chinchilla says ~20 tokens per parameter is compute-optimal.
    optimal = (n_train + n_val) / 20
    print(f"\nAt ~20 tokens/param, this corpus is compute-optimal for a "
          f"~{optimal/1e6:.0f}M parameter model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
