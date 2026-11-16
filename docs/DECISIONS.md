# Decisions

Architecture decision records. Each is dated by the run that introduced it;
superseding an ADR requires a new entry, not an edit to history.

## ADR-0001 — fp16-only, SDPA/eager attention on Turing

**Decision:** train and export in fp16. Reject bf16, FlashAttention 2/3, and
`torch.compile` attention paths that assume `sm_80+`.

**Why:** the runtime is a Tesla T4 (`sm_75`, 14.56 GiB). FlashAttention is not
available; bf16 is not reliably supported by the kernels used here. The env
gate and `state.validate_config` both enforce this, and a static test scans source
and configs for forbidden patterns.

**Consequence:** no bf16 fallback. If a dependency only works in bf16, it is not
usable, regardless of convenience.

## ADR-0002 — fp16 base, fp32 LoRA adapters

**Decision:** keep the frozen base in fp16, cast every trainable parameter
(LoRA A/B) to fp32, and use `GradScaler`.

**Why:** `GradScaler.unscale_` cannot unscale fp16 gradients; the first smoke run
failed with `Attempting to unscale FP16 gradients`. fp32 adapters are small
(8.7k–~80M params), fit easily, and preserve fp16 memory savings for the base.

**Consequence:** adapter checkpoints are fp32. Merged export casts back to fp16.
Do not “optimize” this by casting adapters to fp16 without a different scaler
strategy.

## ADR-0003 — EOS-separated packing for CPT; no packing for SFT/DPO

**Decision:** CPT packs documents into fixed `seq_len` samples separated by EOS.
Standard causal attention is allowed across document boundaries. SFT and DPO
do not pack.

**Why:** packing is necessary to avoid wasting T4 compute on padding for a
domain corpus. Document isolation would require a block-diagonal mask that this
stack does not support efficiently. The contamination is recorded here and in
`docs/DATA.md`, not hidden.

**Consequence:** CPT loss/perplexity are slightly optimistic at boundaries.
Evaluation uses unpacked held-out text. SFT/DPO must use their phase-specific
prompt masking and batching; the CPT loop refuses those phases.

## ADR-0004 — batch split is operational, not run identity

**Decision:** `micro_batch` and `grad_accum` are excluded from the config hash.
Their product with `seq_len` is hashed as `effective_tokens_per_step`.

**Why:** the OOM ladder must be able to halve `micro_batch` and double
`grad_accum` without changing run identity or invalidating checkpoints. The
optimisation trajectory depends on effective tokens per step, not the split.

**Consequence:** changing the split alone resumes the same run. Changing
`lr`, `seq_len`, model, LoRA config, seed, data revision, or effective tokens
does not.

**Residual risk:** a CLI `--max-steps` override is not folded into the hash; it
can change the cosine schedule while keeping the same run id. Treat
`--max-steps` as an operational horizon for smoke/chaos tests, not a way to
silently extend a production run. A schedule-changing extension deserves a new
run id.

## ADR-0005 — content-addressed data revisions, index last

**Decision:** `data_revision = sha256(canonical index-without-created_utc)`,
shards are addressed by `rev-<hash>`, and `index.json` is uploaded after all
shards. `LATEST_DATA` is a convenience pointer only; training pins the hash.

**Why:** rebuilding identical data must not create a new run identity, and a
reader must never see an index whose shards are missing. `created_utc` is
informational and excluded from the hash.

**Consequence:** data is immutable. Fixing a corpus requires a new revision;
never overwrite `rev-*`.

## ADR-0006 — local directory backend is a test transport only

**Decision:** `T4LLM_HUB_ROOT` selects `LocalBackend`. Real runs use the
Hugging Face HTTP backend.

**Why:** the chaos test and CPU suite need atomic commit semantics without a
network or a writable Hub repo. `LocalBackend` uses a staging directory and a
single rename; published checkpoints are immutable.

**Consequence:** local runs are not durable across machine loss. Do not present
them as a production result.

## ADR-0007 — no `huggingface_hub.login()`

**Decision:** read `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` from the environment and
pass it per API call. Never call `login()`, never write a token file.

**Why:** `login()` persists the token under the user's home directory. A
notebook token must not become a file artifact.

**Consequence:** every subprocess must inherit the environment; the supervisor
does. Error messages and reprs never include the token.

## ADR-0008 — standard HF gradient checkpointing instead of unsloth

**Decision:** do not install unsloth in this runtime. Use non-reentrant HF
gradient checkpointing and record the fallback at runtime.

**Why:** the available unsloth release pins `transformers<=5.17.0`, while the
working runtime has 5.22.1. Downgrading transformers risks the rest of the
stack on a live T4. The fallback was smoke-tested.

**Consequence:** somewhat slower/less memory-optimal than unsloth, but a
reproducible install. Revisit only with a measured A/B and a compatible pin.

## ADR-0009 — checkpoint tiers

**Decision:** full checkpoints contain adapter, optimizer, scheduler, all four
RNG states, tokenizer, config, env, trainer state, SHA256SUMS, manifest.
Weights-only checkpoints omit optimizer/scheduler/RNG and trigger an optimizer
reset plus a 20-step LR re-warmup on resume.

**Why:** full checkpoints are larger and slower to upload; weights-only reduces
the worst-case loss window when upload bandwidth is poor. Retention always keeps
at least `keep_last` full checkpoints so a weights-only file is never the only
survivor.

## ADR-0010 — f16 GGUF, no invented quantization

**Decision:** export an f16 GGUF via llama.cpp's `convert_hf_to_gguf.py`.
Quantized variants are a separate, measured step; absent a quantization run, the
repo does not claim one.

**Why:** correctness and reproducibility first. A q4_k_m file that was never
evaluated is not a deliverable.

## ADR-0011 — evaluation must include a frozen-base delta

**Decision:** every eval scores the frozen base and the adapter on identical
chunks. A general set is optional, but if absent the result explicitly says
`not_measured` and does not claim “no regression”.

**Why:** domain perplexity alone can hide catastrophic forgetting. The frozen
base is already resident, so the delta is cheap.

**Consequence:** eval needs a held-out domain JSONL. Training data is never
used as the eval set.

## ADR-0012 — phase trainers share durability, not the CPT loss path

**Decision:** the CPT loop rejects `run.phase` other than `cpt`. SFT and DPO use
a separate trainer (`phase_core.py` plus `sft.py`/`pref.py`) that reuses the
checkpoint, manifest, resume, OOM and push machinery but owns its batching and
loss. SFT masks prompt tokens and supervises only the completion; DPO computes
reference log-probabilities with the LoRA adapter disabled. Packing is refused
for both phases.

**Why:** running SFT through the CPT loss would train on prompt tokens and pack
across conversations. Duplicating the durability code instead of sharing it
would let the two paths drift on the one contract that must never drift.

**Consequence:** `configs/sft.yaml` and `configs/dpo.yaml` are runnable once
`data.sft_file` / `data.pref_file` points at a licence-declaring JSONL. The
`train` CLI still refuses non-CPT phases; use `sft`, `dpo`, or `supervise`.
Do not remove the CPT guard to “unify” the paths.

## ADR-0013 — token/credential handling in logs and state

**Decision:** redact token-like keys from any diagnostic JSON; never print the
environment; store only repo ids and commit metadata in run state.

**Why:** a single leaked notebook output can invalidate a token. The code never
needs the secret after the API client is constructed.

## ADR-0014 — phase data identity is the JSONL file digest

**Decision:** an SFT/DPO run identifies its data by
`file-sha256:<sha256 of the JSONL>` in `run.data_revision`. The cursor stores
`shard_idx = epoch`, `sample_idx = position in the deterministic shuffled
epoch`, and `consumed_tokens = supervised completion tokens`. A changed file is
a different run; resume validates the digest before restoring.

**Why:** the CPT data identity is a packed-shard revision with its own index.
Phase data has no shards and must remain append-friendly, but the same rule
applies: moved data cannot silently continue an old run.

**Consequence:** editing a JSONL invalidates resume by design. Rebuild/repin or
start a new run; never rewrite the file under a live run.

## ADR-0015 — first real run: Python code SFT on Qwen3-1.7B-Base

**Decision (operator delegated “make all the right decisions and start”):**
the first real run is SFT of `Qwen/Qwen3-1.7B-Base` (Apache-2.0) on Python
function-level instruction data from `bigcode/self-oss-instruct-sc2-exec-filter-50k`,
pinned at revision `356bb069eee815daa6e23e9a282eeefe1490ad44`, licensed
**ODC-BY-1.0** (attribution required). Domain: Python code generation.

- Model repo: `Ggboykxz/t4llm-python-sft` (checkpoints + merged export).
- Data repo: `Ggboykxz/t4llm-python-data` (dataset, attribution, `heldout.jsonl`).
- Code licence: Apache-2.0 (`LICENSE`).
- Split: deterministic shuffle, seed 3407; 10,000 train + 500 held-out.
- Measured token stats (300-row sample): median 990 tokens/example, max 1892;
  ~9.9M train tokens; fits `seq_len 2048`. At the measured ~1k tok/s this is
  roughly 3 h of supervised cells; resumable via `kill -9`-safe checkpoints.
- DPO is **not** run: no real preference pairs exist for this domain yet.
- CPT is **not** run first: the corpus is ~10M tokens, below the threshold
  where packing-based pretraining pays off; SFT directly is cheaper.
- Evaluation: held-out perplexity against the frozen base; no pass@1/execution
  claim is made until sandboxed evaluation exists (it does not yet).
- Attribution: every record and the data repo carry `license: ODC-BY-1.0` and
  `source: bigcode/self-oss-instruct-sc2-exec-filter-50k@356bb069`; the repo
  ships `NOTICE` with the dataset URL, revision and licence link.

**Why:** the user delegated the decision set and asked to start. Every input is
verifiable (base licence Apache-2.0, dataset licence ODC-BY-1.0, pinned
revision) and the run fits the T4 and the resumable checkpoint design.

**Consequence:** if the domain is changed later, `configs/sft.yaml` and the
data repo are the only places to touch; the trainers, checkpoint schema and
export path are domain-agnostic.

## ADR-0016 — resume loss check is one-sided; EMA tracks faster

**Decision:** `assert_resume_loss` now fails only when the mean of the ten
post-resume losses exceeds `(1 + resume_loss_tolerance)` × the pre-death loss
EMA. A lower mean is accepted. Separately, the loss EMA decay changed from
`0.98/0.02` to `0.90/0.10`.

**Why:** the two-sided check compared a fast-improving SFT run against a
near-step-1 EMA. On a healthy resume (measured on the real Hub checkpoint
`qwen3-1.7b-python-sft-sft-06787eba` step 8) the next-10 mean was 0.624 vs EMA
0.988 — 37 % *below* EMA, i.e. an improvement — and the runner was refused and
ended in `state.status = "fatal"`. Losses that fall can never indicate a broken
restore; only losses that spike above the EMA can. The old EMA was also too
slow to be useful for telemetry or spike detection.

**Consequence:** resume is possible again for runs with falling loss. The
one-sided check still guards corruption (swapped weights, lost optimizer state,
wrong cursor) whose signature is a loss *higher* than the pre-death EMA. Loss
EMA is not part of the run identity; changing the decay does not fork a run.
