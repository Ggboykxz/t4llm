# Runbook

Operational steps, in order. Every command assumes the repo root and
`PYTHONPATH=src` (the Makefile exports it).

## 0. Contract and confirmations

Before a real run, confirm in writing:

1. The domain and held-out evaluation set.
2. `run.hub_repo` (private model repo, e.g. `Ggboykxz/t4-llm-<domain>`).
3. `run.data_repo` (private dataset repo).
4. The licence of every source. A missing/unknown licence stops the data build.
5. The base model. The current default is `Qwen/Qwen3-1.7B-Base`; a change is a
   new run identity, never a silent swap.

Do not create the repos from this code until those names are confirmed. The
token is never written to disk.

```bash
export HF_TOKEN=...   # Colab secret / process env only
```

## 1. Bootstrap

```bash
make setup
```

This installs the pins in `requirements.lock`, never reinstall torch, fetches
the llama.cpp converter for GGUF (unless `T4LLM_SKIP_LLAMA_CPP=1`), prints the
capability report, and runs the CPU contract suite. If the report does not say
`TRAINING-CAPABLE`, stop.

## 2. Data

Input is JSONL, one document per line:

```json
{"text": "...", "source": "my-corpus-v1", "license": "CC-BY-4.0"}
```

```bash
python -m t4llm data \
  --config configs/cpt.yaml \
  --input /path/to/corpus.jsonl \
  --out data \
  --push
```

The output is content-addressed (`rev-<sha256>`). `index.json` is uploaded last,
so a reader never sees a revision without its index. Copy the printed revision
into `configs/base.yaml`:

```yaml
run:
  data_revision: "<64 hex chars>"
data:
  revision: "<same 64 hex chars>"
```

`docs/DATA.md` is generated with real counts, dedup funnel, sources and
licences. Commit that report with the pinned revision.

Validate before training:

```bash
python - <<'PY'
from t4llm.data import load_index
i = load_index("data")
print(i["data_revision"], i["total_tokens"], len(i["shards"]))
PY
```

## 3. Benchmark

```bash
make bench
```

Writes one row per measured run to `docs/BENCH.md`. `SKIPPED` means not
measured; never edit it into a number. The upload column stays `SKIPPED` until
a writable repo exists:

```bash
python -m t4llm bench --config configs/cpt.yaml --upload-repo <user>/<repo>
```

If VRAM is over budget, lower `micro_batch` in the bench command only; then
record the measured value before changing the config.

## 4. Train

```bash
# Notebook cell: supervisor owns the child and a 20-minute budget.
python -m t4llm supervise --config configs/cpt.yaml --time-budget 20m

# Or run the child directly when you control the process lifetime.
python -m t4llm train --config configs/cpt.yaml --time-budget 6m
```

The child checkpoints at the configured cadence (6 minutes / 60 steps), before
advancing, and pushes. `LATEST` is a separate commit. A `kill -9` leaves at most
one checkpoint interval of work to replay, and the cursor makes replay exact.

What the OOM ladder does:

- `OutOfMemoryError` rewinds to the last committed cursor, halves `micro_batch`,
  recomputes `grad_accum` to keep `effective_tokens_per_step`, records
  `oom_stage`, and retries.
- At `micro_batch=1` it refuses rather than changing `seq_len` silently. A
  sequence-length change requires rebuilt data and a new run identity.

What divergence does:

- Non-finite loss/gradient: rewind and rollback with a halved LR scale.
- 20 consecutive steps above `1.5 ×` pre-step EMA: same.
- More than `resilience.max_rollbacks` (default 3): fatal halt. Inspect; do not
  loop forever.

## 5. Monitor

```bash
python -m t4llm status --config configs/cpt.yaml
python -m t4llm status --config configs/cpt.yaml --json
```

Fields: current step, consumed tokens, loss EMA, tok/s, ETA, last successful
push age, checkpoint count, latest checkpoint, repo bytes, last events.

Backpressure is a hard guard: if no push succeeds for
`push_backpressure_minutes` (15), training pauses and waits instead of
continuing with no durable state. Investigate the Hub/network; do not raise the
limit to “get through” a run.

## 6. Resume

```bash
# Verify without training; downloads and hashes the newest valid checkpoint.
python -m t4llm resume --config configs/cpt.yaml

# Resume training; normally just rerun supervise/train.
python -m t4llm train --config configs/cpt.yaml
```

Resolution walks `LATEST` first, then descending steps, skipping corrupt
candidates and recording them in state/events. A config-hash mismatch is fatal
unless you deliberately pass `--allow-config-change`; that writes an ADR entry,
marks the run forked, and resets the optimizer. Never use it casually.

A weights-only checkpoint resumes weights and cursor but resets optimizer and
scheduler, then re-warms the LR for 20 steps. Prefer a full checkpoint when one
exists.

## 7. Chaos test

```bash
make chaos
```

Builds a tiny model/data fixture, launches training three times, kills each with
`SIGKILL` only after a brand-new checkpoint is durable, and validates:

- only complete checkpoints are visible;
- `LATEST` points at a valid manifest;
- state step is never behind `LATEST`;
- every visible checkpoint passes manifest + SHA-256 validation;
- resume events exist and lost steps are within one checkpoint interval;
- the final checkpoint resolves byte-for-byte.

Set `T4LLM_CHAOS_KEEP=1` to keep artifacts under `/tmp/t4llm-chaos.*`.

## 8. Evaluation

```bash
python -m t4llm eval --config configs/cpt.yaml
```

`eval.domain_set` (or `--domain-set`) is required: a held-out JSONL with `text`,
`messages`, or `prompt`+`response`. `eval.general_set` is optional; if absent,
the report says `not_measured` and `regression=false` (not “no regression
measured” disguised as a pass).

The run measures the frozen base and the adapter on the same chunks. Domain
perplexity and the delta are written to `runs/<run>/eval-<step>.json`, appended
to `docs/EVAL.md`, and uploaded to `evals/step-XXXXXXXX.json`.

A general-set perplexity increase above `eval.regression_tolerance` (default
5%) returns exit code 2. Do not export a model that regressed unless you record
why.

## 9. Export

```bash
python -m t4llm export --config configs/cpt.yaml          # safetensors + GGUF
python -m t4llm export --config configs/cpt.yaml --no-gguf
```

The adapter is merged into the fp16 base, saved with `safe_serialization=True`,
and accompanied by `export_manifest.json`, `SHA256SUMS`, and a model card. GGUF
is f16 (`model-f16.gguf`) via `llama.cpp`'s converter; set `LLAMA_CPP_DIR` if it
is not at `/content/llama.cpp`.

The export is pushed to `exports/step-XXXXXXXX/` and `EXPORT_LATEST` only after
the folder is complete. Re-exporting the same step is idempotent only when the
file hashes match; otherwise it refuses to overwrite.

Before publishing:

- set `export.license` (or leave the card's explicit “do not redistribute”);
- confirm `docs/DATA.md` contains every source licence;
- confirm `docs/EVAL.md` has a row for this step, or state that evaluation is
  still pending.

## 10. Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `HF_TOKEN is not set` | secret not exposed | set the Colab secret; never write it to a file |
| HTTP 401/403 | token lacks write scope | fix the fine-grained token; do not retry blindly |
| HTTP 404 | wrong repo/shard path | check `run.hub_repo`/`run.data_repo` |
| `VRAM ... < 14 GiB` | wrong GPU | use a T4 runtime; CPU is test-only |
| `bf16 is forbidden` | config/runtime drift | restore `train.dtype: fp16` |
| `Attempting to unscale FP16 gradients` | adapter stayed fp16 | `cast_trainable_fp32` must run after adapter load |
| `Found an incompatible version of torchao` | peft/torchao mismatch | install `torchao==0.18.0` (bootstrap does) |
| `no valid checkpoint found` | all candidates corrupt | inspect `resume_events.jsonl`; do not delete the Hub history |
| `data revision mismatch` | index moved without repinning | rebuild or repin; never ignore |
| `checkpoint push failed` | network/Hub outage | training stops; rerun after connectivity returns |
| OOM at `micro_batch=1` | sequence/batch too large | rebuild data with smaller `seq_len` or use a smaller base; log the decision |
