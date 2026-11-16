"""`python -m t4llm <cmd>` (§17)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import env
from .state import (
    ConfigError,
    compute_config_hash,
    load_config,
    make_run_id,
    parse_duration,
)


def _config(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if getattr(args, "data_dir", None):
        overrides.setdefault("data", {})["local_dir"] = args.data_dir
    if getattr(args, "model", None):
        overrides.setdefault("model", {})["base"] = args.model
    return load_config(args.config, overrides)


def cmd_env(args: argparse.Namespace) -> int:
    report = env.print_capability_report()
    if not report.training_capable and os.environ.get("T4LLM_ALLOW_NON_T4") != "1":
        return 1
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_bench

    run_bench(
        args.config,
        model=args.model,
        tiny=args.tiny,
        steps=args.steps,
        seq_len=args.seq_len,
        micro_batch=args.micro_batch,
        upload_repo=args.upload_repo,
    )
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from .data import build_packed, iter_jsonl_docs, publish_data, write_data_report
    from .hub import make_backend
    from .state import atomic_write_json

    cfg = _config(args)
    out = Path(args.out)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    if not args.input:
        raise ConfigError("--input JSONL is required (fields: text, source, license)")
    index = build_packed(
        iter_jsonl_docs(args.input),
        out,
        tokenizer,
        seq_len=int(cfg["train"]["seq_len"]),
        seed=int(cfg["run"]["seed"]),
        max_tokens_per_shard=int(cfg["data"]["max_tokens_per_shard"]),
        near_dedup=not args.no_near_dedup,
        tokenizer_name=cfg["model"]["base"],
    )
    write_data_report(index, Path(args.report_dir))
    print(json.dumps({k: index[k] for k in ("data_revision", "total_samples", "total_tokens")}, indent=2))
    if args.push:
        backend = make_backend(cfg, "dataset")
        rev = publish_data(out, backend)
        print(f"published data revision {rev}")
        print("Pin this in configs/base.yaml: run.data_revision: " + rev)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .loop import train

    return train(
        args.config,
        time_budget=args.time_budget,
        max_steps=args.max_steps,
        run_id=args.run_id,
        allow_config_change=args.allow_config_change,
        allow_synthetic=args.allow_synthetic,
        data_dir=args.data_dir,
        model=args.model,
        device=args.device,
    )


def cmd_supervise(args: argparse.Namespace) -> int:
    from .loop import supervise

    return supervise(
        args.config,
        time_budget=args.time_budget,
        max_steps=args.max_steps,
        allow_config_change=args.allow_config_change,
        allow_synthetic=args.allow_synthetic,
        data_dir=args.data_dir,
        model=args.model,
        train_file=args.train_file,
    )


def cmd_sft(args: argparse.Namespace) -> int:
    from .sft import train_sft

    return train_sft(
        args.config,
        time_budget=args.time_budget,
        max_steps=args.max_steps,
        run_id=args.run_id,
        allow_config_change=args.allow_config_change,
        train_file=args.train_file,
        model=args.model,
        device=args.device,
    )


def cmd_dpo(args: argparse.Namespace) -> int:
    from .pref import train_dpo

    return train_dpo(
        args.config,
        time_budget=args.time_budget,
        max_steps=args.max_steps,
        run_id=args.run_id,
        allow_config_change=args.allow_config_change,
        train_file=args.train_file,
        model=args.model,
        device=args.device,
    )


def cmd_resume(args: argparse.Namespace) -> int:
    from .hub import make_backend
    from .loop import prepare_run, runs_root
    from .resume import resolve_and_download

    cfg = _config(args)
    phase = str(cfg["run"]["phase"])
    if phase == "cpt":
        prepared = prepare_run(
            args.config,
            data_dir=args.data_dir,
            model=args.model,
            run_id=args.run_id,
            allow_synthetic=args.allow_synthetic,
        )
    else:
        from .phase_core import prepare_phase_run

        prepared = prepare_phase_run(
            args.config,
            train_file=args.train_file,
            model=args.model,
            run_id=args.run_id,
            require_t4=False,
        )
    backend = make_backend(prepared.cfg, "model")
    resolved = resolve_and_download(
        backend,
        cache_dir=runs_root() / prepared.run_id / "cache-resume-cli",
        expected_config_hash=prepared.cfg_hash,
        allow_config_change=args.allow_config_change,
        config=prepared.cfg,
    )
    if resolved is None:
        print("no checkpoint found")
        return 1
    print(
        json.dumps(
            {
                "run_id": resolved.manifest.run_id,
                "step": resolved.manifest.global_step,
                "tier": resolved.manifest.tier,
                "consumed_tokens": resolved.manifest.consumed_tokens,
                "cursor": resolved.manifest.cursor,
                "local_dir": str(resolved.local_dir),
                "corrupt_steps": resolved.corrupt_steps,
                "forked": resolved.forked,
                "optimizer_reset": resolved.optimizer_reset,
            },
            indent=2,
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from .hub import PushManager, make_backend

    cfg = _config(args)
    run_id_hint: str | None = None
    if str(cfg["run"]["phase"]) != "cpt":
        # Status must resolve the same run id the phase trainer uses, which
        # folds the training-file digest into run.data_revision.  A missing or
        # unreadable file must not make a read-only status command crash.
        try:
            from .phase_core import prepare_phase_run

            prepared = prepare_phase_run(
                args.config,
                train_file=getattr(args, "train_file", None),
                model=args.model,
                require_t4=False,
            )
            cfg = prepared.cfg
            run_id_hint = prepared.run_id
        except Exception:
            pass
    backend = make_backend(cfg, "model")
    mgr = PushManager(backend)
    st = mgr.status()
    hb = st.get("heartbeat", {}) or {}
    run_id = hb.get("run_id") or st.get("run_id") or run_id_hint
    if not run_id:
        try:
            run_id = make_run_id(cfg, compute_config_hash(cfg))
        except Exception:
            run_id = "unknown"
    step = int(hb.get("step", st.get("global_step", 0)) or 0)
    max_steps = int(cfg["train"]["max_steps"])
    eff = int(cfg["train"]["micro_batch"]) * int(cfg["train"]["grad_accum"]) * int(
        cfg["train"]["seq_len"]
    )
    tok_s = hb.get("tok_s")
    eta = None
    if tok_s and step < max_steps:
        eta = (max_steps - step) * eff / float(tok_s)
    last_push = hb.get("last_push_utc") or st.get("last_push_utc")
    age = None
    if last_push:
        try:
            age = (
                datetime.now(timezone.utc) - datetime.fromisoformat(last_push)
            ).total_seconds()
        except Exception:
            age = None
    events = hb.get("events") or hb.get("last_events") or []
    out = {
        "run_id": run_id,
        "step": step,
        "max_steps": max_steps,
        "tokens_consumed": hb.get("consumed_tokens", st.get("consumed_tokens")),
        "loss_ema": hb.get("loss_ema", st.get("loss_ema")),
        "tok_s": tok_s,
        "eta_seconds": None if eta is None else int(eta),
        "last_push_utc": last_push,
        "last_push_age_seconds": None if age is None else int(age),
        "session_age_seconds": hb.get("session_age_s"),
        "checkpoint_count": len(st.get("checkpoints", [])),
        "latest_checkpoint": st.get("latest"),
        "repo_bytes": st.get("repo_bytes"),
        "last_events": events[-3:],
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"{k:24}: {v}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval import run_eval

    result = run_eval(
        args.config,
        checkpoint_step=args.checkpoint_step,
        domain_set=args.domain_set,
        limit=args.limit,
        skip_general=args.skip_general,
        train_file=args.train_file,
    )
    return 0 if result.get("regression", False) is False else 2


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_model

    export_model(
        args.config,
        checkpoint_step=args.checkpoint_step,
        gguf=not args.no_gguf,
        push=not args.no_push,
        out_dir=args.out,
        train_file=args.train_file,
    )
    return 0


def cmd_fixture(args: argparse.Namespace) -> int:
    from .fixtures import make_tiny_run

    meta = make_tiny_run(
        args.root,
        seq_len=args.seq_len,
        num_samples=args.samples,
        max_steps=args.max_steps,
        ckpt_steps=args.ckpt_steps,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
    )
    print(json.dumps(meta, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="t4llm", description="Crash-only T4 specialization")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("env", help="probe GPU and enforce the capability contract")
    e.set_defaults(func=cmd_env)

    b = sub.add_parser("bench", help="measure tok/s, VRAM, checkpoint size, upload")
    b.add_argument("--config", default="configs/cpt.yaml")
    b.add_argument("--model", default=None)
    b.add_argument("--tiny", action="store_true")
    b.add_argument("--steps", type=int, default=10)
    b.add_argument("--seq-len", type=int, default=None)
    b.add_argument("--micro-batch", type=int, default=None)
    b.add_argument("--upload-repo", default=None)
    b.set_defaults(func=cmd_bench)

    d = sub.add_parser("data", help="dedup, pack, publish immutable shards")
    d.add_argument("--config", default="configs/cpt.yaml")
    d.add_argument("--input", default=None)
    d.add_argument("--out", default="data")
    d.add_argument("--report-dir", default="docs")
    d.add_argument("--push", action="store_true")
    d.add_argument("--no-near-dedup", action="store_true")
    d.add_argument("--data-dir", default=None)
    d.add_argument("--model", default=None)
    d.set_defaults(func=cmd_data)

    for name, fn in (("train", cmd_train), ("supervise", cmd_supervise)):
        t = sub.add_parser(name, help="run/resume the CPT child or its supervisor")
        t.add_argument("--config", default="configs/cpt.yaml")
        t.add_argument("--time-budget", default=None)
        t.add_argument("--max-steps", type=int, default=None)
        t.add_argument("--run-id", default=None)
        t.add_argument("--allow-config-change", action="store_true")
        t.add_argument("--allow-synthetic", action="store_true")
        t.add_argument("--data-dir", default=None)
        t.add_argument("--model", default=None)
        if name == "train":
            t.add_argument("--device", default=None)
        else:
            t.add_argument("--train-file", default=None)
        t.set_defaults(func=fn)

    for name, fn in (("sft", cmd_sft), ("dpo", cmd_dpo)):
        t = sub.add_parser(name, help=f"run/resume the {name.upper()} phase trainer")
        t.add_argument("--config", default=f"configs/{name}.yaml")
        t.add_argument("--train-file", default=None)
        t.add_argument("--time-budget", default=None)
        t.add_argument("--max-steps", type=int, default=None)
        t.add_argument("--run-id", default=None)
        t.add_argument("--allow-config-change", action="store_true")
        t.add_argument("--model", default=None)
        t.add_argument("--device", default=None)
        t.set_defaults(func=fn)

    r = sub.add_parser("resume", help="resolve and verify a checkpoint without training")
    r.add_argument("--config", default="configs/cpt.yaml")
    r.add_argument("--run-id", default=None)
    r.add_argument("--data-dir", default=None)
    r.add_argument("--train-file", default=None)
    r.add_argument("--model", default=None)
    r.add_argument("--allow-config-change", action="store_true")
    r.add_argument("--allow-synthetic", action="store_true")
    r.set_defaults(func=cmd_resume)

    s = sub.add_parser("status", help="read-only status from Hub state")
    s.add_argument("--config", default="configs/cpt.yaml")
    s.add_argument("--json", action="store_true")
    s.add_argument("--data-dir", default=None)
    s.add_argument("--train-file", default=None)
    s.add_argument("--model", default=None)
    s.set_defaults(func=cmd_status)

    ev = sub.add_parser("eval", help="domain + general regression eval")
    ev.add_argument("--config", default="configs/cpt.yaml")
    ev.add_argument("--checkpoint-step", type=int, default=None)
    ev.add_argument("--domain-set", default=None)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--skip-general", action="store_true")
    ev.add_argument("--train-file", default=None)
    ev.set_defaults(func=cmd_eval)

    ex = sub.add_parser("export", help="merge fp16 LoRA, safetensors + GGUF")
    ex.add_argument("--config", default="configs/cpt.yaml")
    ex.add_argument("--checkpoint-step", type=int, default=None)
    ex.add_argument("--out", default=None)
    ex.add_argument("--no-gguf", action="store_true")
    ex.add_argument("--no-push", action="store_true")
    ex.add_argument("--train-file", default=None)
    ex.set_defaults(func=cmd_export)

    f = sub.add_parser("fixture", help="create a tiny model + synthetic shards for tests")
    f.add_argument("--root", default="/tmp/t4llm-fixture")
    f.add_argument("--seq-len", type=int, default=64)
    f.add_argument("--samples", type=int, default=4000)
    f.add_argument("--max-steps", type=int, default=200)
    f.add_argument("--ckpt-steps", type=int, default=10)
    f.add_argument("--micro-batch", type=int, default=1)
    f.add_argument("--grad-accum", type=int, default=1)
    f.set_defaults(func=cmd_fixture)
    return p


def main(argv: list[str] | None = None) -> int:
    env.configure_process_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except env.EnvError as exc:
        print(f"ENVIRONMENT REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        code = getattr(exc, "exit_code", 2)
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return int(code)
