"""Resume protocol: resolve, verify, restore all RNGs, seek by cursor (§9)."""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .hub import Backend, CorruptCheckpointError, HubError, parse_checkpoint_step
from .state import (
    CheckpointManifest,
    ConfigMismatchError,
    MANIFEST_NAME,
    StateError,
    atomic_write_bytes,
    check_config_hash,
    sha256_file,
    utcnow,
    validate_checkpoint_dir,
    write_fork_adr,
)


class ResumeError(StateError):
    pass


class ResumeDiverged(ResumeError):
    pass


@dataclass
class ResolvedCheckpoint:
    manifest: CheckpointManifest
    local_dir: Path
    corrupt_steps: list[int] = field(default_factory=list)
    forked: bool = False
    optimizer_reset: bool = False
    lost_steps: int = 0


def _candidate_steps(backend: Backend) -> tuple[int | None, list[int]]:
    latest: int | None = None
    try:
        text = backend.read_bytes("LATEST").decode("utf-8").strip()
        latest = parse_checkpoint_step(text)
    except (HubError, UnicodeDecodeError):
        latest = None
    steps: set[int] = set()
    for f in backend.list_files("ckpt-"):
        s = parse_checkpoint_step(f.path)
        if s is not None:
            steps.add(s)
    ordered = sorted(steps, reverse=True)
    if latest in steps:
        ordered = [latest] + [s for s in ordered if s != latest]
    return latest, ordered


def _download_candidate(backend: Backend, step: int, dest: Path) -> CheckpointManifest:
    prefix = f"ckpt-{step:08d}"
    raw = backend.read_bytes(f"{prefix}/{MANIFEST_NAME}")
    manifest = CheckpointManifest.from_dict(json.loads(raw))
    if manifest.global_step != step:
        raise CorruptCheckpointError(
            f"manifest step {manifest.global_step} != directory step {step}"
        )
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel, entry in manifest.files.items():
        target = dest / rel
        backend.download_to(f"{prefix}/{rel}", target)
        if target.stat().st_size != entry.bytes or sha256_file(target) != entry.sha256:
            raise CorruptCheckpointError(f"downloaded hash/size mismatch: {rel}")
    atomic_write_bytes(dest / MANIFEST_NAME, raw)
    validate_checkpoint_dir(dest)
    return manifest


def resolve_and_download(
    backend: Backend,
    *,
    cache_dir: str | Path,
    expected_config_hash: str,
    allow_config_change: bool = False,
    adr_path: str | Path = "docs/DECISIONS.md",
    reason: str = "operator passed --allow-config-change",
    config: dict[str, Any] | None = None,
) -> ResolvedCheckpoint | None:
    latest, steps = _candidate_steps(backend)
    if not steps:
        return None
    corrupt: list[int] = []
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for step in steps:
        dest = cache / f"ckpt-{step:08d}"
        try:
            manifest = _download_candidate(backend, step, dest)
        except (HubError, CorruptCheckpointError, StateError, json.JSONDecodeError):
            corrupt.append(step)
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            continue
        if manifest.config_hash != expected_config_hash:
            if not allow_config_change:
                raise ConfigMismatchError(expected_config_hash, manifest.config_hash)
            write_fork_adr(
                adr_path,
                run_id=manifest.run_id,
                old_hash=manifest.config_hash,
                new_hash=expected_config_hash,
                reason=reason,
            )
            return ResolvedCheckpoint(
                manifest=manifest,
                local_dir=dest,
                corrupt_steps=corrupt,
                forked=True,
                optimizer_reset=True,
            )
        return ResolvedCheckpoint(
            manifest=manifest,
            local_dir=dest,
            corrupt_steps=corrupt,
            optimizer_reset=manifest.tier == "weights_only" or manifest.optimizer_reset,
        )
    raise ResumeError(
        f"no valid checkpoint found for run; corrupt candidates: {corrupt or 'none'} "
        f"(LATEST={latest})"
    )


# --------------------------------------------------------------------------- #
# RNG state — all four generators, loadable with weights_only=True
# --------------------------------------------------------------------------- #
def save_rng_state(path: str | Path) -> None:
    py_version, py_state, py_gauss = random.getstate()
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    payload = {
        "python": {
            "version": int(py_version),
            "state": [int(x) for x in py_state],
            "gauss": py_gauss,
        },
        "numpy": {
            "name": str(np_name),
            "keys": [int(x) for x in np_keys],
            "pos": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached": float(np_cached),
        },
        "torch": torch.get_rng_state().clone(),
        "cuda": [t.clone() for t in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_rng_state(path: str | Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    py = payload["python"]
    random.setstate((int(py["version"]), tuple(int(x) for x in py["state"]), py["gauss"]))
    npd = payload["numpy"]
    np.random.set_state(
        (
            str(npd["name"]),
            np.asarray([int(x) for x in npd["keys"]], dtype=np.uint32),
            int(npd["pos"]),
            int(npd["has_gauss"]),
            float(npd["cached"]),
        )
    )
    torch.set_rng_state(payload["torch"].cpu())
    if torch.cuda.is_available() and payload.get("cuda"):
        torch.cuda.set_rng_state_all([t.cpu() for t in payload["cuda"]])


# --------------------------------------------------------------------------- #
# Optimizer / scheduler
# --------------------------------------------------------------------------- #
def save_optimizer(optimizer: torch.optim.Optimizer, path: str | Path) -> None:
    torch.save(optimizer.state_dict(), path)


def load_optimizer(optimizer: torch.optim.Optimizer, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state)


def save_scheduler(scheduler: Any, path: str | Path) -> None:
    torch.save(scheduler.state_dict(), path)


def load_scheduler(scheduler: Any, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    scheduler.load_state_dict(state)


def assert_resume_loss(losses: list[float], ema: float, tolerance: float = 0.05) -> float:
    """§9.7: ten post-resume losses must not exceed (1 + tolerance) × pre-death EMA.

    The check is one-sided. A restored model that *improves* after resume
    cannot indicate a broken restore, so a lower mean loss is accepted. The
    check exists to catch a corrupted or silently swapped restore, whose
    signature is a loss *higher* than the pre-death EMA.
    """
    if len(losses) < 10:
        raise ResumeDiverged(f"need 10 post-resume losses, got {len(losses)}")
    if not ema or ema <= 0:
        return float(np.mean(losses[:10]))
    mean = float(np.mean(losses[:10]))
    if mean > ema * (1.0 + tolerance):
        rel = (mean - ema) / ema
        raise ResumeDiverged(
            f"resume loss check failed: next-10 mean {mean:.4f} vs pre-death EMA "
            f"{ema:.4f} (+{rel * 100:.1f}% > {tolerance * 100:.0f}%); refusing to train"
        )
    return mean


def read_resume_events(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def append_resume_event(path: str | Path, event: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"utc": utcnow(), **event}, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
