"""Run identity, config hashing, checkpoint manifests and run state (§5, §7).

The manifest is a contract, not a log line.  It is written last locally, pulled
first remotely, and every field that determines recovery is validated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TIERS = {"full", "weights_only"}
FULL_REQUIRED = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pt",
    "trainer_state.json",
    "config.json",
    "env.json",
    "SHA256SUMS",
}
WEIGHTS_REQUIRED = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "trainer_state.json",
    "config.json",
    "env.json",
    "SHA256SUMS",
}
ALLOWED_PREFIXES = ("tokenizer/",)
MANIFEST_NAME = "manifest.json"
SUMS_NAME = "SHA256SUMS"


class StateError(RuntimeError):
    pass


class ConfigError(StateError):
    pass


class ManifestError(StateError):
    pass


class ConfigMismatchError(StateError):
    def __init__(self, expected: str, actual: str, message: str = ""):
        self.expected = expected
        self.actual = actual
        super().__init__(
            message
            or f"config_hash mismatch: checkpoint={actual!r} expected={expected!r}; "
            "refusing to resume without --allow-config-change"
        )


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - bootstrap guarantees pyyaml
        raise ConfigError("PyYAML is required; run scripts/bootstrap.sh") from exc
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load base.yaml then the phase file.  Overrides are already nested dicts."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    raw = _read_yaml(p)
    base_path = p.parent / "base.yaml"
    if p.name != "base.yaml" and base_path.exists():
        cfg = deep_merge(_read_yaml(base_path), raw)
    else:
        cfg = raw
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ("run", "model", "lora", "train", "data"):
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"missing config section: {section}")
    train, model, run, data = cfg["train"], cfg["model"], cfg["run"], cfg["data"]
    if (train.get("dtype") or "").lower() != "fp16":
        raise ConfigError("train.dtype must be fp16; bf16 is forbidden on T4")
    attn = (model.get("attn") or "").lower()
    if "flash" in attn:
        raise ConfigError("model.attn must be sdpa (Turing has no FlashAttention)")
    if attn not in {"sdpa", "eager"}:
        raise ConfigError(f"model.attn={attn!r}; allowed sdpa/eager")
    if model.get("load_in_4bit") and _params_b(model.get("base", "")) < 4.0:
        raise ConfigError("model.load_in_4bit requires a >=4B base; use fp16 LoRA below that")
    for key in ("seq_len", "micro_batch", "grad_accum", "max_steps"):
        value = train.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"train.{key} must be a positive integer")
    if int(data.get("seq_len", train["seq_len"])) != int(train["seq_len"]):
        raise ConfigError("data.seq_len must equal train.seq_len")
    eff = train["micro_batch"] * train["grad_accum"] * train["seq_len"]
    if eff < 1024 and os.environ.get("T4LLM_ALLOW_SMALL_BATCH") != "1":
        raise ConfigError(
            f"effective tokens/step is implausibly low: {eff}; "
            "set T4LLM_ALLOW_SMALL_BATCH=1 only for tiny tests"
        )
    phase = run.get("phase", "cpt")
    if phase not in {"cpt", "sft", "dpo"}:
        raise ConfigError(f"unknown phase: {phase!r}")
    if run.get("data_revision") is None and data.get("revision") is not None:
        run["data_revision"] = data["revision"]


def _params_b(model_name: str) -> float:
    """Parse a parameter count from a model id; unknown -> 0 (treated as small)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:[-_]|$)", model_name)
    if not m:
        return 0.0
    return float(m.group(1))


def base_slug(cfg: dict[str, Any]) -> str:
    explicit = cfg.get("run", {}).get("base_slug")
    if explicit:
        return re.sub(r"[^a-z0-9._-]+", "-", str(explicit).lower()).strip("-")
    name = str(cfg.get("model", {}).get("base", "model")).rstrip("/").split("/")[-1]
    name = re.sub(r"-(?:base|it|instruct|chat)$", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")


# Only fields that can change the optimisation trajectory (or the data it sees)
# enter the run identity.  Hub paths and resilience timers are operational.
TRAIN_RELEVANT = {
    "run": {"base_slug", "phase", "seed", "data_revision"},
    "model": {"base", "load_in_4bit", "attn", "gradient_checkpointing", "trust_remote_code"},
    "lora": {"r", "alpha", "dropout", "use_rslora", "target"},
    "train": {
        "dtype",
        "seq_len",
        # micro_batch/grad_accum are intentionally excluded; their product is
        # hashed as effective_tokens_per_step so the OOM ladder can re-split
        # without silently changing the optimisation trajectory (ADR-0004).
        "lr",
        "scheduler",
        "warmup_ratio",
        "weight_decay",
        "max_grad_norm",
        "optim",
        "max_steps",
        "packing",
        "train_on_responses_only",
        "neftune_noise_alpha",
        "num_train_epochs",
        "beta",
        "loss_type",
    },
    "data": {"revision", "seq_len"},
}


def config_for_hash(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for section, keys in TRAIN_RELEVANT.items():
        values = cfg.get(section, {}) or {}
        picked = {k: values[k] for k in keys if k in values}
        if picked:
            out[section] = picked
    t = cfg.get("train", {}) or {}
    if all(k in t for k in ("micro_batch", "grad_accum", "seq_len")):
        out.setdefault("train", {})["effective_tokens_per_step"] = (
            int(t["micro_batch"]) * int(t["grad_accum"]) * int(t["seq_len"])
        )
    return out


def compute_config_hash(cfg: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config_for_hash(cfg)).encode("utf-8"))[:8]


def make_run_id(cfg: dict[str, Any], cfg_hash: str | None = None) -> str:
    h = cfg_hash or compute_config_hash(cfg)
    phase = str(cfg.get("run", {}).get("phase", "cpt"))
    rid = f"{base_slug(cfg)}-{phase}-{h}"
    if not RUN_ID_RE.match(rid):
        raise ConfigError(f"generated invalid run_id: {rid!r}")
    return rid


def check_config_hash(
    checkpoint_hash: str, expected_hash: str, *, allow_config_change: bool = False
) -> None:
    if checkpoint_hash == expected_hash:
        return
    if not allow_config_change:
        raise ConfigMismatchError(expected_hash, checkpoint_hash)
    # The caller must also call write_fork_adr() and mark the run forked.


def write_fork_adr(
    path: str | Path,
    *,
    run_id: str,
    old_hash: str,
    new_hash: str,
    reason: str,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"\n### {utcnow()} — forked `{run_id}`\n"
        f"- old config_hash: `{old_hash}`\n"
        f"- new config_hash: `{new_hash}`\n"
        f"- reason: {reason}\n"
        f"- consequence: checkpoint is loadable only with `--allow-config-change`; "
        f"metrics mark `forked: true` and `optimizer_reset: true`.\n"
    )
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(entry)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    try:
        dfd = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


@dataclass
class Cursor:
    data_revision: str
    shard_idx: int = 0
    sample_idx: int = 0
    consumed_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_revision": self.data_revision,
            "shard_idx": self.shard_idx,
            "sample_idx": self.sample_idx,
            "consumed_tokens": self.consumed_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cursor":
        if not isinstance(d, dict):
            raise ManifestError("cursor must be an object")
        missing = {"data_revision", "shard_idx", "sample_idx", "consumed_tokens"} - set(d)
        if missing:
            raise ManifestError(f"cursor missing fields: {sorted(missing)}")
        if not isinstance(d["data_revision"], str) or not d["data_revision"]:
            raise ManifestError("cursor.data_revision must be a non-empty string")
        for k in ("shard_idx", "sample_idx", "consumed_tokens"):
            if not isinstance(d[k], int) or d[k] < 0:
                raise ManifestError(f"cursor.{k} must be a non-negative integer")
        return cls(
            data_revision=d["data_revision"],
            shard_idx=d["shard_idx"],
            sample_idx=d["sample_idx"],
            consumed_tokens=d["consumed_tokens"],
        )


@dataclass
class FileEntry:
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "bytes": self.bytes}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileEntry":
        if not isinstance(d, dict) or set(d) != {"sha256", "bytes"}:
            raise ManifestError("file entry must be {sha256, bytes}")
        if not isinstance(d["sha256"], str) or not SHA256_RE.match(d["sha256"]):
            raise ManifestError(f"invalid sha256: {d['sha256']!r}")
        if not isinstance(d["bytes"], int) or d["bytes"] < 0:
            raise ManifestError(f"invalid byte count: {d['bytes']!r}")
        return cls(d["sha256"], d["bytes"])


@dataclass
class CheckpointManifest:
    schema_version: int
    run_id: str
    global_step: int
    tier: str
    consumed_tokens: int
    cursor: dict[str, Any]
    config_hash: str
    created_utc: str
    loss_ema: float
    files: dict[str, FileEntry]
    forked: bool = False
    optimizer_reset: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "global_step": self.global_step,
            "tier": self.tier,
            "consumed_tokens": self.consumed_tokens,
            "cursor": self.cursor,
            "config_hash": self.config_hash,
            "created_utc": self.created_utc,
            "loss_ema": self.loss_ema,
            "files": {k: v.to_dict() for k, v in sorted(self.files.items())},
            "forked": self.forked,
            "optimizer_reset": self.optimizer_reset,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointManifest":
        d = migrate_manifest(d)
        required = {
            "schema_version",
            "run_id",
            "global_step",
            "tier",
            "consumed_tokens",
            "cursor",
            "config_hash",
            "created_utc",
            "loss_ema",
            "files",
        }
        missing = required - set(d)
        if missing:
            raise ManifestError(f"manifest missing fields: {sorted(missing)}")
        files_raw = d["files"]
        if not isinstance(files_raw, dict):
            raise ManifestError("manifest.files must be an object")
        files = {str(k): FileEntry.from_dict(v) for k, v in files_raw.items()}
        m = cls(
            schema_version=int(d["schema_version"]),
            run_id=str(d["run_id"]),
            global_step=int(d["global_step"]),
            tier=str(d["tier"]),
            consumed_tokens=int(d["consumed_tokens"]),
            cursor=dict(d["cursor"]),
            config_hash=str(d["config_hash"]),
            created_utc=str(d["created_utc"]),
            loss_ema=float(d["loss_ema"]),
            files=files,
            forked=bool(d.get("forked", False)),
            optimizer_reset=bool(d.get("optimizer_reset", False)),
        )
        m.validate()
        return m

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported schema_version={self.schema_version}; "
                f"this build understands {SCHEMA_VERSION}"
            )
        if not RUN_ID_RE.match(self.run_id):
            raise ManifestError(f"invalid run_id: {self.run_id!r}")
        if self.global_step < 0:
            raise ManifestError("global_step must be >= 0")
        if self.tier not in TIERS:
            raise ManifestError(f"invalid tier: {self.tier!r}")
        if not isinstance(self.config_hash, str) or not re.match(r"^[0-9a-f]{8}$", self.config_hash):
            raise ManifestError(f"invalid config_hash: {self.config_hash!r}")
        if not isinstance(self.created_utc, str) or "T" not in self.created_utc:
            raise ManifestError("created_utc must be ISO-8601")
        cursor = Cursor.from_dict(self.cursor)
        if cursor.consumed_tokens != self.consumed_tokens:
            raise ManifestError(
                "consumed_tokens disagrees with cursor.consumed_tokens: "
                f"{self.consumed_tokens} != {cursor.consumed_tokens}"
            )
        if self.loss_ema != self.loss_ema:  # NaN
            raise ManifestError("loss_ema is NaN")
        required = FULL_REQUIRED if self.tier == "full" else WEIGHTS_REQUIRED
        missing = required - set(self.files)
        if missing:
            raise ManifestError(f"{self.tier} checkpoint missing files: {sorted(missing)}")
        if MANIFEST_NAME in self.files:
            raise ManifestError("manifest.json must not hash itself")
        for name, entry in self.files.items():
            _validate_relative_path(name)
            if not isinstance(entry, FileEntry):
                raise ManifestError(f"invalid file entry for {name}")


def _validate_relative_path(name: str) -> None:
    if not name or name.startswith("/") or ".." in Path(name).parts or "\\" in name:
        raise ManifestError(f"unsafe path in manifest: {name!r}")
    if name == MANIFEST_NAME:
        return
    if name == SUMS_NAME:
        return
    if name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pt",
        "trainer_state.json",
        "config.json",
        "env.json",
    ):
        return
    if any(name.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        return
    raise ManifestError(f"unexpected file in checkpoint: {name!r}")


def migrate_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    """Single migration hook.  Old checkpoints must stay loadable forever."""
    if not isinstance(raw, dict):
        raise ManifestError("manifest must be a JSON object")
    version = raw.get("schema_version")
    if version == SCHEMA_VERSION:
        return raw
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise ManifestError(
            f"checkpoint schema_version={version} is newer than this code "
            f"({SCHEMA_VERSION}); upgrade t4llm, do not guess"
        )
    raise ManifestError(f"no migration path from schema_version={version!r}")
    # Future: if version == 1 and SCHEMA_VERSION == 2: transform and set 2.


def enumerate_payload_files(ckpt_dir: str | Path) -> list[str]:
    root = Path(ckpt_dir)
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel == MANIFEST_NAME:
                continue
            if rel.endswith(".tmp") or "/." in f"/{rel}":
                raise ManifestError(f"temporary file inside checkpoint: {rel}")
            out.append(rel)
    return out


def write_sha256sums(ckpt_dir: str | Path) -> dict[str, FileEntry]:
    root = Path(ckpt_dir)
    names = [n for n in enumerate_payload_files(root) if n != SUMS_NAME]
    entries: dict[str, FileEntry] = {}
    lines: list[str] = []
    for rel in names:
        p = root / rel
        digest = sha256_file(p)
        size = p.stat().st_size
        entries[rel] = FileEntry(digest, size)
        lines.append(f"{digest}  {rel}\n")
    atomic_write_text(root / SUMS_NAME, "".join(lines))
    sums = root / SUMS_NAME
    entries[SUMS_NAME] = FileEntry(sha256_file(sums), sums.stat().st_size)
    return entries


def build_manifest(
    ckpt_dir: str | Path,
    *,
    run_id: str,
    global_step: int,
    tier: str,
    consumed_tokens: int,
    cursor: Cursor | dict[str, Any],
    config_hash: str,
    loss_ema: float,
    forked: bool = False,
    optimizer_reset: bool = False,
    created_utc: str | None = None,
) -> CheckpointManifest:
    """Hash payload, write SHA256SUMS, then write manifest.json last."""
    root = Path(ckpt_dir)
    cursor_d = cursor.to_dict() if isinstance(cursor, Cursor) else dict(cursor)
    entries = write_sha256sums(root)
    m = CheckpointManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        global_step=global_step,
        tier=tier,
        consumed_tokens=consumed_tokens,
        cursor=cursor_d,
        config_hash=config_hash,
        created_utc=created_utc or utcnow(),
        loss_ema=float(loss_ema),
        files=entries,
        forked=forked,
        optimizer_reset=optimizer_reset,
    )
    m.validate()
    atomic_write_json(root / MANIFEST_NAME, m.to_dict())
    return m


def read_manifest(ckpt_dir: str | Path) -> CheckpointManifest:
    p = Path(ckpt_dir) / MANIFEST_NAME
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {p}: {exc}") from exc
    return CheckpointManifest.from_dict(raw)


def validate_checkpoint_dir(ckpt_dir: str | Path) -> CheckpointManifest:
    root = Path(ckpt_dir)
    m = read_manifest(root)
    actual = set(enumerate_payload_files(root)) | {MANIFEST_NAME}
    expected = set(m.files) | {MANIFEST_NAME}
    if actual != expected:
        raise ManifestError(
            f"checkpoint contents mismatch; missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    for rel, entry in m.files.items():
        p = root / rel
        if not p.is_file():
            raise ManifestError(f"missing file: {rel}")
        if p.stat().st_size != entry.bytes:
            raise ManifestError(f"size mismatch for {rel}")
        if sha256_file(p) != entry.sha256:
            raise ManifestError(f"sha256 mismatch for {rel}")
    return m


@dataclass
class RunState:
    run_id: str
    phase: str
    config_hash: str
    global_step: int = 0
    consumed_tokens: int = 0
    cursor: dict[str, Any] = field(default_factory=dict)
    loss_ema: float = 0.0
    skipped_steps: int = 0
    total_steps: int = 0
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_utc: str = field(default_factory=utcnow)
    updated_utc: str = field(default_factory=utcnow)
    status: str = "running"
    optimizer_reset: bool = False
    forked: bool = False
    rollbacks: int = 0
    lr_scale: float = 1.0
    oom_stage: int = 0
    last_push_utc: str | None = None
    last_ckpt_step: int | None = None
    best_metric: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "config_hash": self.config_hash,
            "global_step": self.global_step,
            "consumed_tokens": self.consumed_tokens,
            "cursor": self.cursor,
            "loss_ema": self.loss_ema,
            "skipped_steps": self.skipped_steps,
            "total_steps": self.total_steps,
            "session_id": self.session_id,
            "started_utc": self.started_utc,
            "updated_utc": self.updated_utc,
            "status": self.status,
            "optimizer_reset": self.optimizer_reset,
            "forked": self.forked,
            "rollbacks": self.rollbacks,
            "lr_scale": self.lr_scale,
            "oom_stage": self.oom_stage,
            "last_push_utc": self.last_push_utc,
            "last_ckpt_step": self.last_ckpt_step,
            "best_metric": self.best_metric,
            "events": self.events[-50:],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunState":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - allowed
        if unknown:
            raise StateError(f"run state has unknown fields: {sorted(unknown)}")
        return cls(**d)

    def event(self, name: str, **fields: Any) -> None:
        ev = {"utc": utcnow(), "event": name}
        ev.update(fields)
        self.events.append(ev)
        self.events = self.events[-50:]
        self.updated_utc = utcnow()


def run_state_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "state.json"


def save_run_state(run_dir: str | Path, state: RunState) -> None:
    state.updated_utc = utcnow()
    atomic_write_json(run_state_path(run_dir), state.to_dict())


def load_run_state(run_dir: str | Path) -> RunState | None:
    p = run_state_path(run_dir)
    if not p.exists():
        return None
    try:
        return RunState.from_dict(json.loads(p.read_text()))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise StateError(f"corrupt run state {p}: {exc}") from exc


def write_trainer_state(
    path: str | Path,
    *,
    state: RunState,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "run_id": state.run_id,
        "global_step": state.global_step,
        "consumed_tokens": state.consumed_tokens,
        "cursor": state.cursor,
        "epoch": state.consumed_tokens / max(1, int(config.get("data", {}).get("total_tokens", 0) or 1)),
        "best_metric": state.best_metric,
        "loss_ema": state.loss_ema,
        "skipped_steps": state.skipped_steps,
        "total_steps": state.total_steps,
        "optimizer_reset": state.optimizer_reset,
        "forked": state.forked,
        "rollbacks": state.rollbacks,
        "lr_scale": state.lr_scale,
        "oom_stage": state.oom_stage,
        "session_id": state.session_id,
        "updated_utc": utcnow(),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)


def approx_tokens_per_step(cfg: dict[str, Any]) -> int:
    t = cfg["train"]
    return int(t["micro_batch"]) * int(t["grad_accum"]) * int(t["seq_len"])


def elapsed_seconds(start: float) -> float:
    return max(0.0, time.monotonic() - start)


def parse_duration(text: str | int | float) -> int:
    """'20m', '90s', '2h', or a bare number of minutes -> seconds."""
    if isinstance(text, (int, float)):
        return int(text * 60)
    s = str(text).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smh]?)", s)
    if not m:
        raise ValueError(f"invalid duration: {text!r}")
    value = float(m.group(1))
    unit = m.group(2) or "m"
    return int(value * {"s": 1, "m": 60, "h": 3600}[unit])
