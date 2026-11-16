# Data contract

**No production dataset is pinned in this checkout yet.** The status is blocked
until the operator confirms the domain, sources, and their licences. This file
documents the contract; `python -m t4llm data ...` overwrites it with measured
counts and the real source/licence table.

## Input format

One JSON object per line:

```json
{"text": "...", "source": "corpus-name-v1", "license": "CC-BY-4.0"}
```

Required fields for every document:

| Field | Meaning |
|---|---|
| `text` | UTF-8 document text |
| `source` | stable source identifier; all docs for one source must share a licence |
| `license` | SPDX-ish identifier or a URL; `unknown`, `none`, `n/a`, `unlicensed` are refused |

A missing licence stops the build before any shard is written. The builder
never infers a licence from a dataset name or a model card.

## Pipeline

1. Normalise whitespace and lowercase for exact-duplicate detection.
2. Exact duplicate: SHA-256 of normalised text. Dropped.
3. Near duplicate: 64-permutation MinHash over 5-token shingles, 16 bands ×
   4 rows, Jaccard threshold 0.70. Dropped.
4. Tokenize without special tokens, append EOS to each kept document.
5. Concatenate tokens, drop the incomplete tail, split into fixed `seq_len`
   samples. EOS separates documents; cross-document causal attention is
   accepted for CPT (ADR-0003).
6. Build a deterministic permutation seeded by `seed XOR first-8-bytes(stream
   SHA-256)`. The global order is materialised into the shards, so a cursor is a
   contiguous `(shard_idx, sample_idx)` position and seek is O(1).
7. Write Arrow shards, content-hash their `.arrow` payload, build `index.json`.
8. Publish shards first, `index.json` last, then `LATEST_DATA`.

The revision is:

```
data_revision = sha256(canonical_json(index without data_revision and created_utc))
```

Identical input + tokenizer + seed + seq_len => identical revision, even on a
different machine/day. Any content change => a new revision. A revision is
immutable; never overwrite `rev-<hash>`.

## Index schema

```json
{
  "format": "t4llm-packed-v1",
  "data_revision": "<64 hex>",
  "seq_len": 2048,
  "seed": 3407,
  "tokenizer": "Qwen/Qwen3-1.7B-Base",
  "total_samples": 0,
  "total_tokens": 0,
  "stream_sha256": "<64 hex>",
  "shards": [
    {
      "index": 0,
      "file": "shard-00000",
      "sample_start": 0,
      "samples": 0,
      "tokens": 0,
      "sha256": "<arrow payload hash>",
      "files": {"dataset_info.json": {"sha256": "...", "bytes": 0}}
    }
  ],
  "stats": {
    "raw_docs": 0,
    "exact_dups": 0,
    "near_dups": 0,
    "empty_docs": 0,
    "kept_docs": 0,
    "dropped_tail_tokens": 0,
    "raw_tokens": 0
  },
  "sources": {"source-name": {"license": "...", "docs": 0, "tokens": 0}},
  "synthetic": false,
  "created_utc": "..."
}
```

Validation is strict: shard indices are contiguous, `sample_start` sums to
`total_samples`, `total_tokens == total_samples * seq_len`, every file hash/size
matches, and the recomputed revision matches. A mismatch refuses to train.

## Commands

```bash
# build and publish (needs run.data_repo and HF_TOKEN)
python -m t4llm data --config configs/cpt.yaml --input corpus.jsonl --out data --push

# local-only build
python -m t4llm data --config configs/cpt.yaml --input corpus.jsonl --out data

# pin the printed revision in configs/base.yaml, then verify
python -m t4llm resume --config configs/cpt.yaml
```

`data.local_dir` is for tests and single-machine smoke runs. Production
training uses `run.data_revision` and fetches `rev-<hash>` from the dataset repo,
verifying every shard hash; a corrupt local cache is deleted and re-downloaded.

## Phase data (SFT / DPO)

CPT data is packed into immutable shards. SFT and DPO are different: they read
one JSONL file directly, keep each example separate, and identify the data by
`file-sha256:<sha256 of the file>`. No packing, no shard index, no tokenizer at
build time — tokenization happens in the trainer with the base model's exact
chat template.

SFT records (one of):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "sft-v1", "license": "CC-BY-4.0"}
{"prompt": "...", "response": "...", "source": "sft-v1", "license": "CC-BY-4.0"}
```

DPO records:

```json
{"prompt": "...", "chosen": "...", "rejected": "...", "system": "optional", "source": "dpo-v1", "license": "CC-BY-4.0"}
```

Rules enforced before training starts:

- every record declares a real `license`; `unknown`, `none`, `n/a` and
  `unlicensed` are refused;
- every source must use one consistent licence;
- the last SFT message must be the assistant completion and must be non-empty;
- DPO `chosen` and `rejected` must both be non-empty;
- `data.sft_file` / `data.pref_file` (or `--train-file`) is mandatory;
- if `run.data_revision` is pinned, it must equal the file digest.

Editing the JSONL changes the digest and therefore the run identity. Treat the
file as immutable during a run; append by creating a new revision and a new run,
not by rewriting the file underneath a live cursor.

## Collection and legality checklist

- Record the exact source URL/version, retrieval date, and licence text.
- Keep per-source documents; do not merge sources with different licences.
- Honour robots/ToS and personal-data rules. Do not put private user data in a
  public repo. The model repo is expected to be private.
- Remove PII before packing if the domain requires it; the tooling does not
  detect PII.
- Decontaminate the held-out eval set before publishing; it must not appear in
  the training JSONL.
- Write the licence decision for the *model* separately (`export.license`).
  A dataset licence does not automatically license the weights.
