"""Merge a LoRA checkpoint into a standalone fp16 model, then emit GGUF (§13)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from . import loop
from .hub import FatalHubError, HFBackend, LocalBackend, make_backend
from .modeling import build_model, configure_cuda, load_adapter
from .resume import resolve_and_download
from .state import (
    ConfigError,
    atomic_write_json,
    atomic_write_text,
    compute_config_hash,
    load_config,
    sha256_file,
    utcnow,
)


class ExportError(RuntimeError):
    pass


EXPORT_LATEST = "EXPORT_LATEST"


def _export_backend(cfg: dict[str, Any]):
    local = os.environ.get("T4LLM_HUB_ROOT")
    if local:
        return LocalBackend(local)
    repo = (
        (cfg.get("export") or {}).get("repo")
        or cfg.get("run", {}).get("export_repo")
        or cfg.get("run", {}).get("hub_repo")
    )
    if not repo:
        raise ConfigError(
            "no export repo configured: set export.repo or run.hub_repo "
            "(or T4LLM_HUB_ROOT for a local transport)"
        )
    return HFBackend(repo, repo_type="model")


def _llama_cpp_dir() -> Path:
    explicit = os.environ.get("LLAMA_CPP_DIR")
    candidates = [Path(explicit)] if explicit else []
    candidates += [Path("/content/llama.cpp"), Path.cwd() / "llama.cpp", Path.home() / "llama.cpp"]
    for c in candidates:
        if (c / "convert_hf_to_gguf.py").is_file():
            return c
    raise ExportError(
        "GGUF conversion needs llama.cpp's convert_hf_to_gguf.py; "
        "run scripts/bootstrap.sh or set LLAMA_CPP_DIR, or use --no-gguf"
    )


def convert_to_gguf(model_dir: str | Path, out_path: str | Path) -> Path:
    model_dir = Path(model_dir).resolve()
    out_path = Path(out_path).resolve()
    script = _llama_cpp_dir() / "convert_hf_to_gguf.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        str(model_dir),
        "--outfile",
        str(out_path),
        "--outtype",
        "f16",
    ]
    print("export: " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-4000:]
        raise ExportError(f"GGUF conversion failed (rc={proc.returncode}):\n{tail}")
    if not out_path.is_file():
        raise ExportError(f"GGUF converter did not write {out_path}")
    with open(out_path, "rb") as fh:
        magic = fh.read(4)
    if magic != b"GGUF":
        raise ExportError(f"{out_path} is not a GGUF file (magic={magic!r})")
    return out_path


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in {"export_manifest.json", "SHA256SUMS"}:
            continue
        out[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def _write_model_card(
    out: Path,
    *,
    cfg: dict[str, Any],
    run_id: str,
    step: int,
    manifest: dict[str, Any],
) -> None:
    ecfg = cfg.get("export") or {}
    license_name = ecfg.get("license")
    licence_line = (
        f"`{license_name}`"
        if license_name
        else "**UNSPECIFIED — do not redistribute until the operator sets `export.license`.**"
    )
    card = f"""---
base_model: {cfg['model']['base']}
library_name: transformers
tags:
- t4llm
- fp16
- lora-merged
---

# {run_id} — merged checkpoint

This is a merged fp16 model exported from a resumable t4llm run.

| Field | Value |
|---|---|
| Base model | `{cfg['model']['base']}` |
| Phase | `{cfg['run']['phase']}` |
| Run id | `{run_id}` |
| Global step | {step} |
| Config hash | `{manifest['config_hash']}` |
| Data revision | `{manifest.get('data_revision')}` |
| Tokens consumed | {manifest.get('consumed_tokens'):,} |
| Checkpoint tier | `{manifest.get('tier')}` |
| LoRA merge | fp16, merged with PEFT |
| Exported (UTC) | {manifest['created_utc']} |

## Licence

{licence_line}

Dataset licences are recorded in `docs/DATA.md`; a model licence is a separate
legal decision and is never inferred from the base model.

## Evaluation

No benchmark numbers are embedded in this card unless `docs/EVAL.md` contains a
matching run/step row. Unmeasured is not zero.

## Files

- `*.safetensors` — merged fp16 weights, sharded at 4 GB.
- `model-f16.gguf` — GGUF f16 conversion (when enabled).
- `export_manifest.json` — source checkpoint, per-file SHA-256 and sizes.
"""
    atomic_write_text(out / "README.md", card)


def _write_sums(root: Path) -> None:
    lines = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n")
    atomic_write_text(root / "SHA256SUMS", "".join(lines))


def export_model(
    config_path: str | Path,
    *,
    checkpoint_step: int | None = None,
    gguf: bool = True,
    push: bool = True,
    out_dir: str | Path | None = None,
    train_file: str | Path | None = None,
) -> dict[str, Any]:
    configure_cuda()
    cfg = load_config(config_path)
    if str(cfg["run"]["phase"]) != "cpt":
        from .phase_core import prepare_phase_run

        cfg = prepare_phase_run(
            config_path, train_file=train_file, require_t4=False
        ).cfg
    backend = make_backend(cfg, "model")
    resolved = resolve_and_download(
        backend,
        cache_dir=loop.runs_root() / "_export-cache",
        expected_config_hash=compute_config_hash(cfg),
        config=cfg,
    )
    if resolved is None:
        raise ExportError("no checkpoint found on the Hub")
    if checkpoint_step is not None and resolved.manifest.global_step != checkpoint_step:
        raise ExportError(
            f"requested step {checkpoint_step}, resolved {resolved.manifest.global_step}"
        )
    step = resolved.manifest.global_step
    root = Path(out_dir) if out_dir else loop.repo_root() / "exports" / resolved.manifest.run_id / f"step-{step:08d}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"export: resolving adapter for {resolved.manifest.run_id} step {step}")

    bundle = build_model(cfg, device="cuda" if torch.cuda.is_available() else "cpu", attach_lora=False)
    model = load_adapter(bundle.model, resolved.local_dir)
    model.eval()
    merged = model.merge_and_unload()
    if torch.cuda.is_available():
        merged = merged.half()
    merged.config.use_cache = True
    merged.save_pretrained(
        str(root),
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        cfg["model"]["base"], trust_remote_code=bool(cfg["model"].get("trust_remote_code"))
    )
    tokenizer.save_pretrained(str(root))
    if (root / "model.safetensors.index.json").exists():
        index = json.loads((root / "model.safetensors.index.json").read_text())
        if any("adapter" in k.lower() for k in index.get("weight_map", {})):
            raise ExportError("merged export still contains adapter weights")
    gguf_path = None
    if gguf:
        gguf_path = convert_to_gguf(root, root / "model-f16.gguf")

    manifest = {
        "format": "t4llm-export-v1",
        "created_utc": utcnow(),
        "run_id": resolved.manifest.run_id,
        "global_step": step,
        "tier": resolved.manifest.tier,
        "config_hash": resolved.manifest.config_hash,
        "data_revision": cfg.get("run", {}).get("data_revision"),
        "consumed_tokens": resolved.manifest.consumed_tokens,
        "base_model": cfg["model"]["base"],
        "phase": cfg["run"]["phase"],
        "lora_r": int(cfg["lora"]["r"]),
        "lora_alpha": int(cfg["lora"]["alpha"]),
        "merged_dtype": "fp16",
        "gguf": gguf_path.name if gguf_path else None,
        "source_checkpoint_manifest_sha256": sha256_file(resolved.local_dir / "manifest.json"),
    }
    _write_model_card(
        root,
        cfg=cfg,
        run_id=resolved.manifest.run_id,
        step=step,
        manifest={**manifest, "consumed_tokens": resolved.manifest.consumed_tokens},
    )
    manifest["files"] = _file_manifest(root)
    _write_sums(root)
    atomic_write_json(root / "export_manifest.json", manifest)
    total = sum(v["bytes"] for v in manifest["files"].values())
    print(f"export: wrote {root} ({total/1e6:.1f} MB, {len(manifest['files'])} files)")

    if push:
        try:
            target = _export_backend(cfg)
        except ConfigError:
            if os.environ.get("T4LLM_ALLOW_UNPUSHED_EXPORT") == "1":
                print("export: no repo configured; leaving export local (allowed by env)")
                target = None
            else:
                raise
        if target is not None:
            prefix = f"exports/step-{step:08d}"
            existing = target.list_files(prefix + "/")
            if existing:
                remote = json.loads(target.read_bytes(prefix + "/export_manifest.json"))
                if remote.get("files") == manifest["files"]:
                    print(f"export: {prefix} already exists and matches; not rewriting")
                else:
                    raise ExportError(
                        f"{prefix} already exists with different contents; refusing to overwrite"
                    )
            else:
                target.upload_folder(root, prefix, f"merged export step {step}")
                target.upload_bytes(
                    (prefix + "\n").encode("utf-8"), EXPORT_LATEST, f"export {prefix}"
                )
            print(f"export: published {prefix}")
    return {
        "root": str(root),
        "step": step,
        "run_id": resolved.manifest.run_id,
        "gguf": str(gguf_path) if gguf_path else None,
        "bytes": total,
        "files": len(manifest["files"]),
    }
