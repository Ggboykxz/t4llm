"""Held-out domain evaluation and general-regression guard (§12, §14).

Nothing here fabricates a number: if a set is missing the result says
``not_measured`` and the regression flag stays false.  Domain perplexity is
always measured against the same frozen base to make the delta interpretable.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch

from .hub import make_backend
from . import loop
from .modeling import build_model, configure_cuda, load_adapter
from .resume import resolve_and_download
from .state import ConfigError, atomic_write_text, compute_config_hash, load_config, utcnow

MAX_TOKENS_PER_CHUNK = 4096


class EvalError(RuntimeError):
    pass


def _read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"eval set not found: {p}")
    out: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(rec, dict):
                raise EvalError(f"{p}:{lineno}: record must be an object")
            if not any(k in rec for k in ("text", "messages", "prompt", "response")):
                raise EvalError(
                    f"{p}:{lineno}: need one of text/messages/prompt+response"
                )
            out.append(rec)
            if limit and len(out) >= limit:
                break
    if not out:
        raise EvalError(f"eval set is empty: {p}")
    return out


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(enc, "keys"):
        enc = enc["input_ids"]
    return [int(x) for x in enc]


def _encode(tokenizer: Any, rec: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) with -100 on prompt/padding positions."""
    if "messages" in rec:
        ids = _apply_chat_template(tokenizer, rec["messages"])
        return ids, list(ids)
    if "prompt" in rec and "response" in rec:
        prompt_ids = tokenizer(
            str(rec["prompt"]), add_special_tokens=False
        )["input_ids"]
        response_ids = tokenizer(
            str(rec["response"]), add_special_tokens=False
        )["input_ids"]
        ids = list(prompt_ids) + list(response_ids)
        labels = [-100] * len(prompt_ids) + list(response_ids)
        return ids, labels
    ids = tokenizer(str(rec["text"]), add_special_tokens=False)["input_ids"]
    return list(ids), list(ids)


@torch.no_grad()
def _score(
    model: Any,
    tokenizer: Any,
    records: Iterable[dict[str, Any]],
    *,
    seq_len: int,
    device: str,
) -> dict[str, Any]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    docs = 0
    chunks = 0
    for rec in records:
        ids, labels = _encode(tokenizer, rec)
        if not ids:
            continue
        docs += 1
        # Fixed-size non-overlapping windows.  The model's own causal shift is
        # used, so labels are not pre-shifted here.
        for start in range(0, len(ids), seq_len):
            chunk_ids = ids[start : start + seq_len]
            chunk_labels = labels[start : start + seq_len]
            if len(chunk_ids) < 2:
                continue
            kept = sum(1 for x in chunk_labels if x != -100)
            if kept == 0:
                continue
            input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
            label_ids = torch.tensor([chunk_labels], dtype=torch.long, device=device)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.float16,
                enabled=device.startswith("cuda"),
            ):
                out = model(input_ids=input_ids, labels=label_ids)
            loss = float(out.loss.detach().float().cpu())
            if not math.isfinite(loss):
                raise EvalError("non-finite eval loss")
            total_nll += loss * kept
            total_tokens += kept
            chunks += 1
    if total_tokens == 0:
        raise EvalError("eval set produced no scored tokens")
    mean_nll = total_nll / total_tokens
    return {
        "documents": docs,
        "chunks": chunks,
        "scored_tokens": total_tokens,
        "nll": mean_nll,
        "perplexity": math.exp(min(50.0, mean_nll)),
    }


def _resolve(config: dict[str, Any], step: int | None):
    backend = make_backend(config, "model")
    resolved = resolve_and_download(
        backend,
        cache_dir=loop.runs_root() / "_eval-cache" / (str(step) if step else "latest"),
        expected_config_hash=compute_config_hash(config),
        config=config,
    )
    if resolved is None:
        raise EvalError("no checkpoint found on the Hub")
    if step is not None and resolved.manifest.global_step != step:
        raise EvalError(
            f"requested checkpoint step {step}, resolved {resolved.manifest.global_step}"
        )
    return backend, resolved


def _append_report(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        atomic_write_text(
            path,
            "# Evaluation\n\n"
            "Measured on held-out JSONL only. `not_measured` is written rather "
            "than a guessed value when no general set is configured.\n\n"
            "| UTC | run | step | domain ppl | base domain ppl | general ppl | "
            "base general ppl | regression | regression tolerance |\n"
            "|---|---|---:|---:|---:|---:|---:|---|---:|\n",
        )
    def cell(v: Any) -> str:
        return "not_measured" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)

    line = (
        f"| {row['utc']} | {row['run_id']} | {row['step']} | {cell(row['domain_ppl'])} | "
        f"{cell(row['base_domain_ppl'])} | {cell(row['general_ppl'])} | "
        f"{cell(row['base_general_ppl'])} | {row['regression']} | "
        f"{row['regression_tolerance']} |\n"
    )
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def run_eval(
    config_path: str | Path,
    *,
    checkpoint_step: int | None = None,
    domain_set: str | None = None,
    limit: int | None = None,
    skip_general: bool = False,
    device: str | None = None,
    train_file: str | Path | None = None,
) -> dict[str, Any]:
    configure_cuda()
    cfg = load_config(config_path)
    if str(cfg["run"]["phase"]) != "cpt":
        from .phase_core import prepare_phase_run

        cfg = prepare_phase_run(
            config_path, train_file=train_file, require_t4=False
        ).cfg
    ecfg = cfg.get("eval", {}) or {}
    domain_path = domain_set or ecfg.get("domain_set")
    if not domain_path:
        raise ConfigError(
            "eval.domain_set is not configured; point it at a held-out JSONL "
            "(fields: text or messages or prompt+response) before running eval"
        )
    limit = int(limit if limit is not None else ecfg.get("limit") or 0) or None
    domain_records = _read_jsonl(domain_path, limit)
    general_path = ecfg.get("general_set") or os.environ.get("T4LLM_GENERAL_SET")
    general_records: list[dict[str, Any]] = []
    if general_path and not skip_general:
        g_samples = int(ecfg.get("general_samples", 512))
        general_records = _read_jsonl(general_path, min(g_samples, limit or g_samples))
    elif not skip_general:
        print(
            "eval: no eval.general_set/T4LLM_GENERAL_SET configured; "
            "general regression is NOT measured"
        )

    backend, resolved = _resolve(cfg, checkpoint_step)
    run_id = resolved.manifest.run_id
    step = resolved.manifest.global_step
    seq_len = min(int(cfg["train"]["seq_len"]), MAX_TOKENS_PER_CHUNK)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu" and os.environ.get("T4LLM_ALLOW_NON_T4") != "1":
        raise EvalError("CPU eval requires T4LLM_ALLOW_NON_T4=1 (tests only)")

    print(f"eval: building frozen base {cfg['model']['base']}")
    bundle = build_model(cfg, device=dev, attach_lora=False)
    model = bundle.model
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        cfg["model"]["base"], trust_remote_code=bool(cfg["model"].get("trust_remote_code"))
    )
    if not tokenizer.eos_token_id:
        raise EvalError("tokenizer has no eos_token_id")
    base_domain = _score(
        model, tokenizer, domain_records, seq_len=seq_len, device=dev
    )
    base_general = (
        _score(model, tokenizer, general_records, seq_len=seq_len, device=dev)
        if general_records
        else None
    )

    print(f"eval: loading adapter {resolved.local_dir} (step {step})")
    tuned = load_adapter(model, resolved.local_dir)
    tuned.eval()
    domain = _score(tuned, tokenizer, domain_records, seq_len=seq_len, device=dev)
    general = (
        _score(tuned, tokenizer, general_records, seq_len=seq_len, device=dev)
        if general_records
        else None
    )

    tolerance = float(ecfg.get("regression_tolerance", 0.05))
    regression = False
    if general is not None and base_general is not None:
        regression = general["perplexity"] > base_general["perplexity"] * (1.0 + tolerance)
    result = {
        "utc": utcnow(),
        "run_id": run_id,
        "step": step,
        "tier": resolved.manifest.tier,
        "config_hash": resolved.manifest.config_hash,
        "data_revision": cfg.get("run", {}).get("data_revision"),
        "domain_set": str(domain_path),
        "general_set": str(general_path) if general_records else None,
        "domain_ppl": domain["perplexity"],
        "base_domain_ppl": base_domain["perplexity"],
        "domain_delta": domain["perplexity"] - base_domain["perplexity"],
        "general_ppl": None if general is None else general["perplexity"],
        "base_general_ppl": None if base_general is None else base_general["perplexity"],
        "regression": regression,
        "regression_tolerance": tolerance,
        "domain": domain,
        "general": general,
        "base_domain": base_domain,
        "base_general": base_general,
        "notes": [] if general is not None else ["general regression not measured"],
    }
    out_path = loop.runs_root() / run_id / f"eval-{step:08d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    _append_report(loop.repo_root() / "docs" / "EVAL.md", result)
    # Eval results are diagnostics, not training state; keep them separately.
    try:
        backend.upload_bytes(
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            f"evals/step-{step:08d}.json",
            f"eval step {step}",
        )
    except Exception as exc:  # eval should not fail merely because upload did
        print(f"eval: warning: result upload failed: {exc}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
