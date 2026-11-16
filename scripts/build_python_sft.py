#!/usr/bin/env python
"""Build the Python code SFT dataset used by the first real run (ADR-0015).

Reads `bigcode/self-oss-instruct-sc2-exec-filter-50k` at a pinned revision and
emits a deterministic train/held-out split:

    data/python-sft/train.jsonl    {prompt, response, source, license}
    data/python-sft/heldout.jsonl {text, source, license}   (eval: `text` only)
    data/python-sft/NOTICE         attribution and licence
    data/python-sft/MANIFEST.json  counts, revision, sha256 of both JSONL files

Usage:
    python scripts/build_python_sft.py [--out data/python-sft] [--n-train 10000] [--n-heldout 500]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
REVISION = "356bb069eee815daa6e23e9a282eeefe1490ad44"
PARQUET = "data/train-00000-of-00001.parquet"
SOURCE = f"{DATASET}@{REVISION[:8]}"
LICENSE = "ODC-BY-1.0"

NOTICE = f"""Attribution (ODC-BY-1.0 — attribution required)
=================================================

This dataset is derived from:

  {DATASET}
  revision: {REVISION}
  file:     {PARQUET}
  licence:  Open Data Commons Attribution License (ODC-BY-1.0)
  url:      https://huggingface.co/datasets/{DATASET}

  Wei, Y., Cassano, F., Liu, J. et al. "Knowledge Transfer from High-Resource
  to Low-Resource Programming Languages for Code LLMs." arXiv:2308.09895.

Every record in train.jsonl / heldout.jsonl carries
  source:  {SOURCE}
  license: {LICENSE}

The derived split (deterministic shuffle, seed 3407; 10,000 train + 500
held-out examples) is marked in MANIFEST.json with per-file SHA-256.

No modifications to the content: prompt and response are copied verbatim.
The `concepts`, `seed`, `sha1`, `id` and `fingerprint` columns are dropped.
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/python-sft")
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    parquet = hf_hub_download(
        DATASET,
        PARQUET,
        repo_type="dataset",
        revision=REVISION,
    )
    df = pd.read_parquet(parquet)

    order = list(range(len(df)))
    random.Random(args.seed).shuffle(order)
    need = args.n_train + args.n_heldout
    train_idx, held_idx = order[: args.n_train], order[args.n_train : need]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_path, held_path = out / "train.jsonl", out / "heldout.jsonl"

    with open(train_path, "w", encoding="utf-8") as fh:
        for i in train_idx:
            row = df.iloc[i]
            rec = {
                "prompt": str(row["prompt"]),
                "response": str(row["response"]),
                "source": SOURCE,
                "license": LICENSE,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(held_path, "w", encoding="utf-8") as fh:
        for i in held_idx:
            row = df.iloc[i]
            rec = {
                "text": str(row["prompt"]) + "\n\n" + str(row["response"]),
                "source": SOURCE,
                "license": LICENSE,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (out / "NOTICE").write_text(NOTICE, encoding="utf-8")
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "split_seed": args.seed,
        "n_train": len(train_idx),
        "n_heldout": len(held_idx),
        "train_sha256": sha256_file(train_path),
        "heldout_sha256": sha256_file(held_path),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
