"""Measured throughput/VRAM/checkpoint/upload numbers (§1, §11)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import env
from .hub import HFBackend, LocalBackend, probe_upload
from .loop import _save_checkpoint, repo_root
from .modeling import (
    build_model,
    build_optimizer,
    build_scheduler,
    configure_cuda,
    set_seed,
)
from .state import Cursor, RunState, atomic_write_text, load_config, utcnow


def _human_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _append_bench(row: dict[str, Any]) -> Path:
    p = repo_root() / "docs" / "BENCH.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        atomic_write_text(
            p,
            "# Benchmarks\n\n"
            "All numbers below are measured by `python -m t4llm bench` on the actual "
            "runtime. Unmeasured rows are marked SKIPPED, never guessed.\n\n"
            "| UTC | GPU | torch | model | mode | seq | micro-bs | tok/s | VRAM peak GiB | "
            "ckpt bytes | upload s | upload Mbps | notes |\n"
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n",
        )
    line = (
        f"| {row['utc']} | {row['gpu']} | {row['torch']} | {row['model']} | "
        f"{row['mode']} | {row['seq']} | {row['micro_batch']} | {row['tok_s']} | "
        f"{row['vram_gib']} | {row['ckpt_bytes']} | {row['upload_s']} | "
        f"{row['upload_mbps']} | {row['notes']} |\n"
    )
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line)
    return p


def run_bench(
    config_path: str | Path,
    *,
    model: str | None = None,
    tiny: bool = False,
    steps: int = 10,
    seq_len: int | None = None,
    micro_batch: int | None = None,
    upload_repo: str | None = None,
) -> dict[str, Any]:
    configure_cuda()
    cfg = load_config(config_path)
    if model:
        cfg["model"]["base"] = model
    from .state import _params_b

    report = env.require_t4(
        dtype=cfg["train"]["dtype"],
        attn=cfg["model"]["attn"],
        load_in_4bit=False,
        model_params_b=_params_b(cfg["model"]["base"]) or None,
    )
    tmp_root = Path(tempfile.mkdtemp(prefix="t4llm-bench-"))
    try:
        if tiny:
            from .fixtures import make_tiny_model

            tiny_dir = make_tiny_model(tmp_root / "tiny", seq_len=seq_len or 64)
            cfg["model"]["base"] = str(tiny_dir)
            cfg["lora"]["r"] = 4
            cfg["lora"]["alpha"] = 8
            seq_len = seq_len or 64
            micro_batch = micro_batch or 2
        seq_len = int(seq_len or cfg["train"]["seq_len"])
        micro_batch = int(micro_batch or cfg["train"]["micro_batch"])
        cfg["train"]["seq_len"] = seq_len
        cfg["train"]["micro_batch"] = micro_batch
        cfg["train"]["grad_accum"] = 1
        set_seed(int(cfg["run"]["seed"]))
        bundle = build_model(cfg)
        model_obj = bundle.model
        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
            cfg["model"]["base"]
        )
        opt, _ = build_optimizer(model_obj, cfg, allow_fallback=True)
        sched = build_scheduler(opt, cfg, max(1, steps + 2))
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        vocab = int(model_obj.config.vocab_size)
        x = torch.randint(0, max(1, vocab - 2), (micro_batch, seq_len), device="cuda")

        # Warmup, then timed steps.  Upload/push is not part of tok/s.
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model_obj(input_ids=x, labels=x)
            scaler.scale(out.loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        for _ in range(max(1, steps)):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model_obj(input_ids=x, labels=x)
            scaler.scale(out.loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.monotonic() - t0
        tok_s = (steps * micro_batch * seq_len) / max(1e-9, elapsed)
        vram = (
            torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        )

        state = RunState(
            run_id="bench-run-00000000",
            phase="bench",
            config_hash="00000000",
            global_step=0,
            cursor=Cursor(data_revision="bench", consumed_tokens=0).to_dict(),
        )
        ckpt_root = tmp_root / "ckpt"
        ckpt_root.mkdir()
        ckpt_path = _save_checkpoint(
            run_dir=ckpt_root,
            step=0,
            tier="full",
            state=state,
            cfg=cfg,
            model=model_obj,
            optimizer=opt,
            scheduler=sched,
            cursor=Cursor(data_revision="bench", consumed_tokens=0),
            loss_ema=0.0,
            forked=False,
            optimizer_reset=False,
            runtime_notes=bundle.notes,
            env_report=report.to_dict(),
            tokenizer=tokenizer,
        )
        ckpt_bytes = _dir_bytes(ckpt_path)

        upload_s: float | None = None
        upload_mbps: str = "SKIPPED"
        notes = "; ".join(bundle.notes) or ""
        if os.environ.get("T4LLM_HUB_ROOT"):
            backend = LocalBackend(os.environ["T4LLM_HUB_ROOT"])
            upload_s = probe_upload(backend)
            upload_mbps = f"{2*1024*1024*8/max(1e-9, upload_s)/1e6:.2f} (local)"
            notes = (notes + "; upload probe is local, not network").strip("; ")
        elif upload_repo:
            backend = HFBackend(upload_repo, repo_type="model")
            upload_s = probe_upload(backend)
            upload_mbps = f"{2*1024*1024*8/max(1e-9, upload_s)/1e6:.2f}"
        else:
            notes = (notes + "; upload not measured (no --upload-repo/HF_TOKEN)").strip("; ")

        row = {
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu": report.gpu_name,
            "torch": report.torch_version,
            "model": cfg["model"]["base"],
            "mode": "LoRA fp16 r%d" % int(cfg["lora"]["r"]),
            "seq": seq_len,
            "micro_batch": micro_batch,
            "tok_s": round(tok_s, 1),
            "vram_gib": round(vram, 2),
            "ckpt_bytes": ckpt_bytes,
            "upload_s": "SKIPPED" if upload_s is None else round(upload_s, 2),
            "upload_mbps": upload_mbps,
            "notes": notes,
        }
        path = _append_bench(row)
        print(json.dumps({**row, "bench_file": str(path)}, indent=2))
        print(f"checkpoint bytes: {_human_bytes(ckpt_bytes)}")
        return row
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
