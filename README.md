# t4llm — crash-only domain specialization on one free T4

A small, resumable, Hub-backed pipeline for specializing a 0.6B–1.7B causal LM
on a single Tesla T4 (15 GiB, `sm_75`), shipped as merged fp16 safetensors and
GGUF. The durable source of truth is the Hub; local disk is a cache.

## Status (read this first)

| Area | State |
|---|---|
| Environment gate, fp16/SDPA policy | implemented and tested |
| Run identity, config hash, manifests | implemented and tested |
| Atomic Hub publish, verify, retention, retry, backpressure | implemented and tested |
| Immutable data revisions, dedup, packing, cursor | implemented and tested |
| Crash-only CPT loop, OOM ladder, divergence rollback | implemented and tested |
| Resume with RNG/optimizer/scheduler and loss guard | implemented and tested |
| Held-out + general-regression eval | implemented and tested on the tiny fixture |
| Merged safetensors export + GGUF conversion | merge/safetensors tested on the tiny fixture; f16 GGUF conversion measured on Qwen3-1.7B-Base (base, not a fine-tuned export) |
| SFT trainer | implemented and tested: prompt masking, no packing, NEFTune, crash-only resume |
| DPO trainer | implemented and tested: sigmoid objective, adapter-disabled reference, crash-only resume |
| Real domain run | **blocked** on domain, repo names, dataset licence, and base confirmation |

Nothing in this repository invents a benchmark, a licence, or a training result.
Unmeasured rows say `not_measured` or `SKIPPED`.

## Non-negotiables enforced in code

- `fp16` only. `bf16` is hard-rejected; FlashAttention is hard-rejected on `sm_75`.
- The frozen base stays fp16; trainable LoRA adapters are fp32 so `GradScaler`
  can unscale gradients (ADR-0002).
- A checkpoint is a content-addressed directory. `manifest.json` is written
  last locally and read first remotely. Partial directories are never visible.
- `LATEST` moves only after the checkpoint directory is uploaded and verified,
  in a separate commit.
- Training advances only after the checkpoint is durable (backpressure guard).
- `micro_batch`/`grad_accum` may be re-split by the OOM ladder without changing
  run identity; `effective_tokens_per_step` is hashed instead (ADR-0004).
- Every data revision is a SHA-256 of its content; the index is uploaded last.
- Missing licence = refused data build. No exceptions.
- The HF token is read from `HF_TOKEN` per call. It is never logged, persisted,
  or passed to `huggingface_hub.login()`.

## Runtime

Measured on this host: Tesla T4, 14.56 GiB, compute capability 7.5, Python
3.13.16, torch 2.14.1+cu130, transformers 5.22.1, peft 0.21.2, trl 1.18.0,
bitsandbytes 0.50.2, datasets 4.8.5. See `docs/BENCH.md` for the measured
throughput/VRAM/checkpoint/upload table.

Source mirrors:

- GitHub: <https://github.com/Ggboykxz/t4llm>
- Hugging Face source backup: <https://huggingface.co/Ggboykxz/t4llm-source>
  (source only, no weights or data; no licence chosen)

## Quick start

```bash
make setup        # bootstrap, pinned deps, capability report, CPU tests
export HF_TOKEN=...   # Colab secret; never write this to a file
make bench        # measured T4 numbers; upload row is SKIPPED until a repo exists
make data         # requires --input JSONL; see docs/DATA.md
make test         # CPU contract suite
make chaos        # kill -9 continuity test on the tiny fixture
make push         # GitHub source mirror
make backup       # HF source backup (archive + git bundle)
```

Fill in `run.hub_repo`, `run.data_repo`, and the pinned `run.data_revision`
before a real run. Training refuses to start without a pinned revision and a
writable model repo.

```bash
# supervised child, time-budgeted for a notebook cell
python -m t4llm supervise --config configs/cpt.yaml --time-budget 20m
python -m t4llm status --config configs/cpt.yaml
python -m t4llm resume --config configs/cpt.yaml
python -m t4llm eval --config configs/cpt.yaml
python -m t4llm export --config configs/cpt.yaml

# phase 2/3: point data.sft_file / data.pref_file at a licence-declaring JSONL
python -m t4llm sft --config configs/sft.yaml --train-file sft.jsonl
python -m t4llm dpo --config configs/dpo.yaml --train-file dpo.jsonl
```

`supervise` dispatches CPT, SFT and DPO from `run.phase`; it relaunches the
child after crashes, stops on fatal resume/config errors, and relies on the
wall-clock budget rather than signal handlers. The child can be `kill -9`ed at
any point.

## Layout

```
configs/            base + CPT/SFT/DPO YAML
src/t4llm/
  env.py            GPU probe and hard policy gates
  state.py          run identity, manifests, atomic writes
  hub.py            local/HF backends, verify, retention, backpressure
  data.py           deterministic dedup/pack, immutable revisions, cursor
  modeling.py       fp16 base + fp32 LoRA, optimizer/scheduler
  resume.py         resolution, verification, RNG/optimizer restore
  loop.py           CPT loop and supervising parent
  phase_core.py     shared SFT/DPO durability, cursor and phase data identity
  sft.py            prompt-masked SFT (no packing, NEFTune)
  pref.py           sigmoid DPO with adapter-disabled reference
  eval.py           held-out perplexity and general-regression guard
  export.py         merge fp16 LoRA, safetensors, GGUF
  bench.py          measured benchmark writer
  fixtures.py       tiny model/data for tests and chaos
scripts/            bootstrap.sh, chaos_test.sh, push_hf_backup.py
tests/              CPU contract suite + GPU end-to-end tests
notebooks/run.ipynb end-to-end notebook: setup, data, CPT/SFT/DPO, eval, export
docs/               RUNBOOK, DECISIONS, DATA, BENCH, EVAL
```

## Hub layout

Model repo:

```
ckpt-00000120/       immutable full checkpoint (adapter, optimizer, RNG, manifest)
ckpt-00000180/       immutable weights-only checkpoint (optimizer reset on resume)
LATEST               separate commit; resolves to ckpt-XXXXXXXX
heartbeat.json       <=1 KB progress heartbeat
metrics.jsonl        append-only local metrics, latest copy uploaded
evals/step-...json   eval diagnostics
exports/step-.../    merged safetensors + GGUF + export manifest
EXPORT_LATEST
```

Dataset repo:

```
rev-<sha256>/shard-00000/...   immutable Arrow shards
rev-<sha256>/index.json        uploaded last
LATEST_DATA                    human convenience pointer
```

## Security

- Put the token in a Colab secret named `HF_TOKEN`; the code only reads the
  process environment.
- Fine-grained tokens should have write access only to the two run repos.
- Do not commit `runs/`, `exports/`, `data/`, or any `.env`.
- `.gitignore` already excludes checkpoints, safetensors, GGUF, and local runs.

## Licence

Code is unlicensed until the operator chooses one. Model and dataset licences
are separate decisions. `export.license` is deliberately unset, so the model
card says “UNSPECIFIED — do not redistribute” until you set it.
