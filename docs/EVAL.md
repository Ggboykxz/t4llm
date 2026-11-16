# Evaluation

No evaluation has been run for a production checkpoint yet. The table is
written by `python -m t4llm eval` and only contains measured held-out results.
An absent general set is reported as `not_measured`, not as “no regression”.

| UTC | run | step | domain ppl | base domain ppl | general ppl | base general ppl | regression | regression tolerance |
|---|---|---:|---:|---:|---:|---:|---|---:|

## Phase checkpoints (SFT / DPO)

The eval command resolves the run identity from `run.phase`. For SFT or DPO,
pass the same training file used for the run:

```bash
python -m t4llm eval --config configs/sft.yaml --train-file sft.jsonl \
  --domain-set heldout.jsonl
```

The file digest becomes `run.data_revision`; omitting `--train-file` (when the
config has no `data.sft_file`/`data.pref_file`) is a config-identity mismatch,
not a reason to fall back to the CPT hash. The frozen-base/adapter comparison
and `not_measured` rules are unchanged.
