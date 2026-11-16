"""Crash-only harness shared by the SFT and DPO phase trainers.

The CPT loop is intentionally not reused: it packs documents and computes loss
on every token.  SFT and DPO need prompt masking, no cross-example packing and
(for DPO) a reference policy.  What *is* reused is the durability contract:
the same checkpoint schema, manifest, Hub publication, resume guard, OOM ladder
and divergence policy.

Data identity for a phase is the SHA-256 of the JSONL file, recorded in the
cursor as ``file-sha256:<digest>`` and folded into the config hash through
``run.data_revision``.  A changed file is therefore a different run, never a
silent continuation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from . import env
from .data import DataError, DataExhausted, _require_license
from .hub import BackpressureError, HubError, PushManager, make_backend
from .loop import (
    DivergenceStop,
    FatalResume,
    OOMRefused,
    T4TrainError,
    _acquire_lock,
    _apply_lr_scale,
    _load_tokenizer,
    _meter,
    _read_rollback_marker,
    _save_checkpoint,
    _scaled_step,
    log,
    run_dir_for,
)
from .modeling import (
    assert_no_bf16,
    build_model,
    build_optimizer,
    build_scheduler,
    cast_trainable_fp32,
    configure_cuda,
    load_adapter,
    set_seed,
)
from .resume import (
    ResumeDiverged,
    append_resume_event,
    assert_resume_loss,
    load_optimizer,
    load_rng_state,
    load_scheduler,
    resolve_and_download,
)
from .state import (
    ConfigError,
    ConfigMismatchError,
    Cursor,
    RunState,
    compute_config_hash,
    load_config,
    load_run_state,
    make_run_id,
    parse_duration,
    save_run_state,
    utcnow,
)


PHASE_KEYS = {"sft": "sft_file", "dpo": "pref_file"}
PHASE_NAMES = {"sft": "SFT", "dpo": "DPO"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class PhaseDataset:
    """Ordered, licence-gated JSONL records with a deterministic epoch order."""

    phase = ""

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        seed: int,
        epochs: int,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ConfigError(f"{self.phase}: training file not found: {self.path}")
        raw = self.path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError(f"{self.path} is not valid UTF-8: {exc}") from exc
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.revision = f"file-sha256:{self.sha256}"
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.epochs = max(1, int(epochs))
        self.records: list[dict[str, Any]] = []
        self.license_counts: dict[str, int] = {}
        self.source_counts: dict[str, dict[str, Any]] = {}
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{self.path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(rec, dict):
                raise ConfigError(f"{self.path}:{lineno}: record must be an object")
            self.validate_record(rec, lineno)
            try:
                licence = _require_license(str(rec.get("license", "")))
            except DataError as exc:
                raise ConfigError(f"{self.path}:{lineno}: {exc}") from exc
            source = str(rec.get("source") or self.path.name)
            self.license_counts[licence] = self.license_counts.get(licence, 0) + 1
            src = self.source_counts.setdefault(
                source, {"license": licence, "records": 0}
            )
            if src["license"] != licence:
                raise ConfigError(
                    f"{self.path}:{lineno}: source {source!r} has conflicting "
                    f"licences ({src['license']!r} vs {licence!r})"
                )
            src["records"] += 1
            self.records.append(rec)
        if not self.records:
            raise ConfigError(f"{self.phase}: {self.path} contains no records")
        self._perm_epoch = -1
        self._perm: np.ndarray | None = None

    # -- overridden by phase modules ------------------------------------- #
    def validate_record(self, rec: dict[str, Any], lineno: int) -> None:
        if not isinstance(rec.get("license"), str):
            raise ConfigError(
                f"{self.path}:{lineno}: missing 'license' (a licence is mandatory)"
            )

    def make_batcher(self, tokenizer: Any) -> "PhaseBatcher":
        raise NotImplementedError

    # -- deterministic order --------------------------------------------- #
    def permutation(self, epoch: int) -> np.ndarray:
        if epoch < 0 or epoch >= self.epochs:
            raise DataExhausted(f"{self.phase}: epoch {epoch} outside [0,{self.epochs})")
        if self._perm_epoch != epoch:
            self._perm = np.random.default_rng(self.seed + epoch).permutation(
                len(self.records)
            ).astype(np.int64)
            self._perm_epoch = epoch
        assert self._perm is not None
        return self._perm

    def record_at(self, epoch: int, pos: int) -> dict[str, Any]:
        return self.records[int(self.permutation(epoch)[pos])]

    def cursor(self, epoch: int, pos: int, consumed_tokens: int) -> Cursor:
        return Cursor(
            data_revision=self.revision,
            shard_idx=int(epoch),
            sample_idx=int(pos),
            consumed_tokens=int(consumed_tokens),
        )

    def validate_cursor(self, cursor: Cursor) -> None:
        if cursor.data_revision != self.revision:
            raise FatalResume(
                f"{self.phase}: checkpoint data revision {cursor.data_revision!r} "
                f"!= current file {self.revision!r}"
            )
        if cursor.shard_idx > self.epochs:
            raise FatalResume(
                f"{self.phase}: cursor epoch {cursor.shard_idx} > epochs {self.epochs}"
            )
        if cursor.shard_idx == self.epochs:
            if cursor.sample_idx != 0:
                raise FatalResume(
                    f"{self.phase}: cursor is past all epochs but sample_idx="
                    f"{cursor.sample_idx}; refusing ambiguous resume"
                )
        elif cursor.sample_idx > len(self.records):
            raise FatalResume(
                f"{self.phase}: cursor sample_idx {cursor.sample_idx} > "
                f"{len(self.records)} records"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "file": str(self.path),
            "sha256": self.sha256,
            "data_revision": self.revision,
            "records": len(self.records),
            "epochs": self.epochs,
            "seq_len": self.seq_len,
            "licenses": dict(sorted(self.license_counts.items())),
            "sources": {k: dict(v) for k, v in sorted(self.source_counts.items())},
        }


class PhaseBatcher:
    """Turns a cursor into a padded microbatch with prompt tokens masked."""

    def __init__(self, dataset: PhaseDataset, tokenizer: Any, seq_len: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        if self.seq_len <= 1:
            raise ConfigError("train.seq_len must be > 1")
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if self.pad_id is None:
            raise ConfigError("tokenizer has neither pad_token_id nor eos_token_id")

    def next_batch(self, cursor: Cursor, batch_size: int) -> tuple[dict[str, Any], Cursor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        encoded: list[dict[str, Any]] = []
        epoch = cursor.shard_idx
        pos = cursor.sample_idx
        consumed = int(cursor.consumed_tokens)
        n = len(self.dataset.records)
        while len(encoded) < batch_size:
            if epoch >= self.dataset.epochs:
                if not encoded:
                    raise DataExhausted(
                        f"{self.dataset.phase}: all {self.dataset.epochs} epoch(s) consumed"
                    )
                break
            if pos >= n:
                epoch += 1
                pos = 0
                continue
            rec = self.dataset.record_at(epoch, pos)
            pos += 1
            item = self.encode(rec)
            if item is None:
                continue
            consumed += int(item["supervised_tokens"])
            encoded.append(item)
        batch = self.collate(encoded)
        batch["supervised_tokens"] = int(
            sum(int(item["supervised_tokens"]) for item in encoded)
        )
        return batch, self.dataset.cursor(epoch, pos, consumed)

    def encode(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError

    def collate(self, encoded: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    # -- tokenizer helpers ----------------------------------------------- #
    def _chat_ids(self, messages: list[dict[str, str]], *, generation: bool) -> list[int]:
        enc = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation,
        )
        # transformers returns a BatchEncoding (a Mapping, not a plain dict)
        # even for one conversation.
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], list):
            enc = enc[0]
        return [int(x) for x in enc]

    def _text_ids(self, text: str) -> list[int]:
        enc = self.tokenizer(text, add_special_tokens=False)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    def _prompt_completion(
        self, prompt: str, completion: str, *, system: str | None = None
    ) -> tuple[list[int], list[int]] | None:
        """Return (prompt_ids, completion_ids) using the exact chat template.

        Falls back to raw tokenisation for tokenizers without a chat template.
        The completion is what gets supervised; the prompt is never trained on.
        """
        if getattr(self.tokenizer, "chat_template", None):
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            prompt_ids = self._chat_ids(messages, generation=True)
            full_ids = self._chat_ids(
                messages + [{"role": "assistant", "content": completion}],
                generation=False,
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ConfigError(
                    "chat template cannot be split deterministically into prompt "
                    "and completion; refusing to guess which tokens are supervised"
                )
            completion_ids = full_ids[len(prompt_ids) :]
        else:
            prompt_ids = self._text_ids(prompt)
            completion_ids = self._text_ids(completion)
            if self.tokenizer.eos_token_id is not None:
                completion_ids.append(int(self.tokenizer.eos_token_id))
        if not completion_ids:
            return None
        return prompt_ids, completion_ids

    @staticmethod
    def _truncate(
        prompt_ids: list[int], completion_ids: list[int], seq_len: int
    ) -> tuple[list[int], list[int]] | None:
        """Keep the whole (truncated) completion; drop prompt tokens from the left."""
        completion = list(completion_ids[:seq_len])
        if not completion:
            return None
        room = seq_len - len(completion)
        prompt = list(prompt_ids)
        if len(prompt) > room:
            prompt = prompt[-room:] if room > 0 else []
        return prompt + completion, [-100] * len(prompt) + completion

    def _pad(
        self, rows: list[tuple[list[int], list[int]]]
    ) -> dict[str, Any]:
        width = max(len(ids) for ids, _ in rows)
        input_ids = torch.full(
            (len(rows), width), int(self.pad_id), dtype=torch.long
        )
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, (ids, lbl) in enumerate(rows):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, : len(lbl)] = torch.tensor(lbl, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
@dataclass
class PreparedPhase:
    cfg: dict[str, Any]
    dataset: PhaseDataset
    cfg_hash: str
    run_id: str


def _phase_file_key(phase: str) -> str:
    try:
        return PHASE_KEYS[phase]
    except KeyError as exc:
        raise ConfigError(f"unknown phase {phase!r}; expected sft or dpo") from exc


def _new_dataset(phase: str, cfg: dict[str, Any], path: str | Path) -> PhaseDataset:
    train = cfg["train"]
    common = dict(
        seq_len=int(train["seq_len"]),
        seed=int(cfg["run"]["seed"]),
        epochs=int(train.get("num_train_epochs", 1) or 1),
    )
    if phase == "sft":
        from .sft import SFTDataset

        return SFTDataset(path, **common)
    from .pref import DPODataset

    return DPODataset(path, **common)


def load_phase_config(
    config_path: str | Path,
    *,
    train_file: str | Path | None = None,
    model: str | None = None,
    require_t4: bool = False,
) -> PreparedPhase:
    """Load a phase config and fold the training-file digest into its identity."""
    overrides: dict[str, Any] = {}
    if model:
        overrides.setdefault("model", {})["base"] = model
    cfg = load_config(config_path, overrides)
    phase = str(cfg["run"]["phase"])
    if phase not in PHASE_KEYS:
        raise ConfigError(
            f"phase {phase!r} is not an SFT/DPO phase; use `t4llm train` for CPT"
        )
    key = _phase_file_key(phase)
    path = Path(train_file) if train_file else cfg.get("data", {}).get(key)
    if path is None or str(path) in {"", "None"}:
        raise ConfigError(
            f"no {phase} training file: set data.{key} or pass --train-file"
        )
    dataset = _new_dataset(phase, cfg, path)
    pinned = cfg["run"].get("data_revision") or cfg["data"].get("revision")
    if pinned and str(pinned) != dataset.revision:
        raise ConfigError(
            f"configured data revision {pinned!r} != {phase} file digest "
            f"{dataset.revision!r}; refusing to train on moved data"
        )
    cfg["run"]["data_revision"] = dataset.revision
    cfg["data"]["revision"] = dataset.revision
    cfg_hash = compute_config_hash(cfg)
    rid = make_run_id(cfg, cfg_hash)
    if require_t4:
        from .state import _params_b

        env.require_t4(
            dtype=cfg["train"]["dtype"],
            attn=cfg["model"]["attn"],
            load_in_4bit=bool(cfg["model"].get("load_in_4bit")),
            model_params_b=_params_b(cfg["model"]["base"]) or None,
        )
    return PreparedPhase(cfg=cfg, dataset=dataset, cfg_hash=cfg_hash, run_id=rid)


def prepare_phase_run(
    config_path: str | Path,
    *,
    train_file: str | Path | None = None,
    model: str | None = None,
    run_id: str | None = None,
    require_t4: bool = True,
) -> PreparedPhase:
    prepared = load_phase_config(
        config_path, train_file=train_file, model=model, require_t4=require_t4
    )
    if run_id:
        prepared.run_id = run_id
    return prepared


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_phase(
    phase: str,
    config_path: str | Path,
    *,
    time_budget: str | int | float | None = None,
    max_steps: int | None = None,
    run_id: str | None = None,
    allow_config_change: bool = False,
    train_file: str | Path | None = None,
    model: str | None = None,
    device: str | None = None,
) -> int:
    configure_cuda()
    prepared = prepare_phase_run(
        config_path,
        train_file=train_file,
        model=model,
        run_id=run_id,
        require_t4=True,
    )
    cfg, dataset = prepared.cfg, prepared.dataset
    if str(cfg["run"]["phase"]) != phase:
        raise ConfigError(
            f"config phase {cfg['run']['phase']!r} does not match trainer {phase!r}"
        )
    run_dir = run_dir_for(prepared.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data.json").write_text(
        json.dumps(dataset.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock = _acquire_lock(run_dir)
    try:
        return _train_phase_inner(
            phase=phase,
            cfg=cfg,
            dataset=dataset,
            run_id=prepared.run_id,
            cfg_hash=prepared.cfg_hash,
            run_dir=run_dir,
            time_budget=time_budget,
            max_steps=max_steps,
            allow_config_change=allow_config_change,
            device=device,
        )
    finally:
        try:
            lock.close()
        except Exception:
            pass


def _train_phase_inner(
    *,
    phase: str,
    cfg: dict[str, Any],
    dataset: PhaseDataset,
    run_id: str,
    cfg_hash: str,
    run_dir: Path,
    time_budget: str | int | float | None,
    max_steps: int | None,
    allow_config_change: bool,
    device: str | None,
) -> int:
    start_mono = time.monotonic()
    session_id = uuid.uuid4().hex[:12]
    budget_seconds = (
        parse_duration(time_budget)
        if time_budget is not None
        else int(cfg["resilience"]["time_budget_minutes"]) * 60
    )
    stop_at = start_mono + (budget_seconds - 90 if budget_seconds >= 180 else budget_seconds * 0.60)
    hard_deadline = start_mono + budget_seconds
    max_steps = int(max_steps if max_steps is not None else cfg["train"]["max_steps"])
    if max_steps <= 0:
        raise ConfigError("max_steps must be positive")

    backend = make_backend(cfg, "model")
    push = PushManager(
        backend,
        keep_last=int(cfg["resilience"]["keep_last"]),
        keep_milestones_every=int(cfg["resilience"]["keep_milestones_every"]),
        backpressure_minutes=float(cfg["resilience"]["push_backpressure_minutes"]),
    )
    push.mark_started()

    previous_state = load_run_state(run_dir)
    if previous_state is not None and previous_state.config_hash != cfg_hash:
        raise ConfigMismatchError(cfg_hash, previous_state.config_hash)
    resolved = resolve_and_download(
        backend,
        cache_dir=run_dir / "cache",
        expected_config_hash=cfg_hash,
        allow_config_change=allow_config_change,
        config=cfg,
    )
    if resolved is None and previous_state is not None and previous_state.global_step > 0:
        raise FatalResume(
            "local state says training advanced but the Hub has no valid checkpoint; "
            "refusing to silently restart from step 0"
        )
    if resolved is not None and resolved.manifest.run_id != run_id:
        raise FatalResume(
            f"checkpoint run_id {resolved.manifest.run_id!r} != expected {run_id!r}"
        )

    if resolved is not None:
        start_step = resolved.manifest.global_step
        cursor = Cursor.from_dict(resolved.manifest.cursor)
        dataset.validate_cursor(cursor)
        loss_ema = float(resolved.manifest.loss_ema)
        optimizer_reset = resolved.optimizer_reset or resolved.forked
        forked = resolved.forked
        lost_steps = max(
            0, (previous_state.global_step if previous_state else start_step) - start_step
        )
        state = previous_state or RunState(
            run_id=run_id, phase=phase, config_hash=cfg_hash
        )
        state.phase = phase
        state.global_step = start_step
        state.cursor = cursor.to_dict()
        state.consumed_tokens = cursor.consumed_tokens
        state.loss_ema = loss_ema
        state.session_id = session_id
        state.status = "running"
        state.config_hash = cfg_hash
        state.event(
            "resumed",
            from_step=start_step,
            lost_steps=lost_steps,
            corrupt_steps=resolved.corrupt_steps,
            optimizer_reset=optimizer_reset,
        )
        append_resume_event(
            run_dir / "resume_events.jsonl",
            {
                "event": "resumed",
                "from_step": start_step,
                "lost_steps": lost_steps,
                "corrupt_steps": resolved.corrupt_steps,
                "optimizer_reset": optimizer_reset,
                "forked": forked,
                "phase": phase,
            },
        )
        for bad in resolved.corrupt_steps:
            state.event("corrupt_checkpoint", step=bad)
        log(
            f"resuming {phase} {run_id} at step {start_step} (lost {lost_steps}, "
            f"optimizer_reset={optimizer_reset}, corrupt={resolved.corrupt_steps})"
        )
    else:
        start_step = 0
        cursor = dataset.cursor(0, 0, 0)
        loss_ema = 0.0
        optimizer_reset = False
        forked = False
        lost_steps = 0
        state = RunState(run_id=run_id, phase=phase, config_hash=cfg_hash)
        state.cursor = cursor.to_dict()
        state.event(
            "started",
            max_steps=max_steps,
            data_revision=dataset.revision,
            records=len(dataset.records),
            epochs=dataset.epochs,
        )
        log(f"starting fresh {phase} run {run_id}")

    state.session_id = session_id
    save_run_state(run_dir, state)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        try:
            metrics_path.write_bytes(backend.read_bytes("metrics.jsonl"))
        except HubError:
            metrics_path.touch()

    committed = Cursor(
        cursor.data_revision, cursor.shard_idx, cursor.sample_idx, cursor.consumed_tokens
    )

    set_seed(int(cfg["run"]["seed"]) + (start_step if optimizer_reset else 0))
    log(f"building {cfg['model']['base']} (fp16 LoRA, {cfg['model']['attn']}, {phase})")
    bundle = build_model(cfg, device=device, attach_lora=resolved is None)
    model = bundle.model
    runtime_notes = list(bundle.notes)
    if resolved is not None:
        model = load_adapter(model, resolved.local_dir)
    model.train()
    on_cuda = torch.cuda.is_available() and (device is None or device.startswith("cuda"))
    if on_cuda:
        cast_trainable_fp32(model)
    assert_no_bf16(model)
    tokenizer = _load_tokenizer(cfg)
    batcher = dataset.make_batcher(tokenizer)
    push.mark_started()

    opt, opt_notes = build_optimizer(model, cfg, allow_fallback=not on_cuda)
    runtime_notes.extend(opt_notes)
    runtime_notes.append(
        "phase trainer: prompt/completion masking, no packing, "
        f"data_revision={dataset.revision}"
    )
    scheduler = build_scheduler(opt, cfg, max_steps)

    if resolved is not None and not optimizer_reset:
        try:
            load_optimizer(opt, resolved.local_dir / "optimizer.pt")
            load_scheduler(scheduler, resolved.local_dir / "scheduler.pt")
            load_rng_state(resolved.local_dir / "rng_state.pt")
            log("optimizer, scheduler and all four RNG states restored")
        except Exception as exc:
            raise FatalResume(f"failed to restore full checkpoint state: {exc}") from exc
    elif resolved is not None:
        state.optimizer_reset = True
        state.event("optimizer_reset", reason="weights_only/forked")
        log("optimizer reset; scheduler rebuilt; 20-step LR re-warmup")

    marker = _read_rollback_marker(run_dir / "rollback.json")
    if marker:
        scale = float(marker.get("scale", 0.5))
        count = int(marker.get("count", state.rollbacks + 1))
        if count > int(cfg["resilience"]["max_rollbacks"]):
            state.status = "fatal"
            state.event("divergence_halt", rollbacks=count)
            save_run_state(run_dir, state)
            raise DivergenceStop(f"{count} divergence rollbacks; halting per contract")
        if scale < state.lr_scale:
            state.lr_scale = scale
        state.rollbacks = max(state.rollbacks, count)
        state.event("divergence_rollback", scale=state.lr_scale, count=state.rollbacks)
        save_run_state(run_dir, state)
        (run_dir / "rollback.json").unlink(missing_ok=True)
    if state.lr_scale != 1.0:
        _apply_lr_scale(opt, state.lr_scale)

    reset_warmup_remaining = 20 if (resolved is not None and optimizer_reset) else 0
    total_reset_warmup = 20

    scaler = torch.amp.GradScaler("cuda", enabled=on_cuda)
    tcfg = cfg["train"]
    micro_batch = int(tcfg["micro_batch"])
    grad_accum = int(tcfg["grad_accum"])
    seq_len = int(tcfg["seq_len"])
    effective_tokens = micro_batch * grad_accum * seq_len
    ckpt_minutes = float(cfg["resilience"]["ckpt_every_minutes"])
    ckpt_steps = int(cfg["resilience"]["ckpt_every_steps"])
    heartbeat_seconds = int(cfg["resilience"]["heartbeat_seconds"])
    last_ckpt_time = time.monotonic()
    last_ckpt_step = start_step
    last_full_time = time.monotonic()
    last_heartbeat = 0.0
    resume_check: list[float] | None = (
        [] if (resolved is not None and not forked and not optimizer_reset) else None
    )
    consecutive_spikes = 0
    env_report = env.capability_report()
    session_start = time.time()

    def trigger_rollback(reason: str) -> None:
        if state.rollbacks + 1 > int(cfg["resilience"]["max_rollbacks"]):
            state.status = "fatal"
            state.event("divergence_halt", rollbacks=state.rollbacks + 1, reason=reason)
            save_run_state(run_dir, state)
            raise DivergenceStop(f"{reason}: divergence rollback limit reached")
        from .state import atomic_write_json

        atomic_write_json(
            run_dir / "rollback.json",
            {"scale": 0.5 ** (state.rollbacks + 1), "count": state.rollbacks + 1},
        )
        state.status = "rollback"
        state.event("divergence_detected", step=state.global_step, reason=reason)
        save_run_state(run_dir, state)
        raise DivergenceStop(
            f"{reason} at step {state.global_step}; rollback to previous checkpoint"
        )

    def checkpoint(tier: str, *, force: bool = False) -> None:
        nonlocal last_ckpt_time, last_ckpt_step, last_full_time
        if state.global_step == last_ckpt_step and not force:
            return
        path = _save_checkpoint(
            run_dir=run_dir,
            step=state.global_step,
            tier=tier,
            state=state,
            cfg=cfg,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            cursor=committed,
            loss_ema=loss_ema,
            forked=forked,
            optimizer_reset=state.optimizer_reset,
            runtime_notes=runtime_notes,
            env_report=env_report,
            tokenizer=tokenizer,
        )
        save_run_state(run_dir, state)
        try:
            if not push.submit(path, step=state.global_step, metrics_path=metrics_path):
                push.wait()
                push.submit(path, step=state.global_step, metrics_path=metrics_path)
        except Exception as exc:
            raise T4TrainError(f"checkpoint push failed: {exc}") from exc
        last_ckpt_time = time.monotonic()
        last_ckpt_step = state.global_step
        if tier == "full":
            last_full_time = time.monotonic()
        log(f"checkpoint queued at step {state.global_step} ({tier})")
        _meter(metrics_path, {"event": "checkpoint", "step": state.global_step, "tier": tier})

    log(
        f"run={run_id} phase={phase} step={start_step} max_steps={max_steps} "
        f"budget={budget_seconds}s effective_tokens={effective_tokens} "
        f"records={len(dataset.records)} data={dataset.revision}"
    )
    if resolved is not None:
        _meter(
            metrics_path,
            {
                "event": "resume",
                "step": start_step,
                "resumed_from_step": start_step,
                "lost_steps": lost_steps,
                "optimizer_reset": state.optimizer_reset,
                "session_id": session_id,
                "elapsed": 0.0,
            },
            fsync=True,
        )

    status = "stopped"
    exit_code = 0
    try:
        while state.global_step < max_steps:
            if time.monotonic() >= stop_at:
                status = "stopped"
                log("time budget reached; checkpointing and exiting 0")
                break
            try:
                push.check_backpressure()
            except BackpressureError as exc:
                log(f"backpressure: {exc}; waiting for the in-flight push")
                push.wait()
                push.check_backpressure()

            opt.zero_grad(set_to_none=True)
            accumulated = 0
            step_losses: list[float] = []
            step_extra: dict[str, list[float]] = {}
            step_start = time.monotonic()
            tokens_this_step = 0
            try:
                while accumulated < grad_accum:
                    if time.monotonic() >= hard_deadline:
                        break
                    try:
                        batch, batch_cursor = batcher.next_batch(cursor, micro_batch)
                    except DataExhausted:
                        status = "data_exhausted"
                        log("data exhausted; stopping rather than repeating silently")
                        break
                    if on_cuda:
                        for key, value in list(batch.items()):
                            if torch.is_tensor(value):
                                batch[key] = value.to("cuda", non_blocking=True)
                    try:
                        with torch.autocast(
                            device_type="cuda" if on_cuda else "cpu",
                            dtype=torch.float16,
                            enabled=on_cuda,
                        ):
                            loss, extra = compute_phase_loss(
                                phase, model, batch, cfg, on_cuda=on_cuda
                            )
                    except torch.cuda.OutOfMemoryError:
                        raise
                    if loss is None or not torch.isfinite(loss):
                        cursor = Cursor(
                            committed.data_revision,
                            committed.shard_idx,
                            committed.sample_idx,
                            committed.consumed_tokens,
                        )
                        trigger_rollback("non-finite loss")
                    scaler.scale(loss / grad_accum).backward()
                    accumulated += 1
                    cursor = batch_cursor
                    tokens_this_step += int(batch.get("supervised_tokens", 0))
                    step_losses.append(float(loss.detach().float().cpu()))
                    for k, v in extra.items():
                        step_extra.setdefault(k, []).append(float(v))
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                cursor = Cursor(
                    committed.data_revision,
                    committed.shard_idx,
                    committed.sample_idx,
                    committed.consumed_tokens,
                )
                opt.zero_grad(set_to_none=True)
                if micro_batch > 1:
                    micro_batch = max(1, micro_batch // 2)
                    grad_accum = max(1, effective_tokens // (micro_batch * seq_len))
                    state.oom_stage += 1
                    state.event(
                        "oom_ladder",
                        stage=state.oom_stage,
                        micro_batch=micro_batch,
                        grad_accum=grad_accum,
                        effective_tokens=micro_batch * grad_accum * seq_len,
                    )
                    save_run_state(run_dir, state)
                    log(
                        f"OOM: micro_batch -> {micro_batch}, grad_accum -> {grad_accum} "
                        f"(effective tokens unchanged)"
                    )
                    continue
                state.status = "fatal"
                state.event("oom_refused", message=str(exc)[:300])
                save_run_state(run_dir, state)
                raise OOMRefused(
                    "OOM at micro_batch=1; seq_len reduction requires rebuilt data and "
                    "would change the effective batch, so refusing rather than guessing"
                ) from exc

            if accumulated == 0:
                if status == "data_exhausted":
                    break
                cursor = Cursor(
                    committed.data_revision,
                    committed.shard_idx,
                    committed.sample_idx,
                    committed.consumed_tokens,
                )
                break
            if accumulated < grad_accum:
                opt.zero_grad(set_to_none=True)
                cursor = Cursor(
                    committed.data_revision,
                    committed.shard_idx,
                    committed.sample_idx,
                    committed.consumed_tokens,
                )
                break

            scale_before = scaler.get_scale()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(tcfg["max_grad_norm"]),
            )
            finite = bool(torch.isfinite(grad_norm)) and all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            )
            if finite:
                scaler.step(opt)
                scaler.update()
                _scaled_step(
                    scheduler,
                    opt,
                    lr_scale=state.lr_scale,
                    reset_warmup_remaining=reset_warmup_remaining,
                )
                if reset_warmup_remaining > 0:
                    reset_warmup_remaining -= 1
                state.global_step += 1
                state.total_steps += 1
                committed = cursor
                state.cursor = committed.to_dict()
                state.consumed_tokens = committed.consumed_tokens
                mean_loss = float(np.mean(step_losses))
                previous_ema = loss_ema
                if loss_ema <= 0:
                    loss_ema = mean_loss
                else:
                    loss_ema = 0.90 * loss_ema + 0.10 * mean_loss
                state.loss_ema = float(loss_ema)
                if previous_ema > 0 and mean_loss > 1.5 * previous_ema:
                    consecutive_spikes += 1
                    state.event(
                        "loss_spike",
                        step=state.global_step,
                        loss=mean_loss,
                        ema=previous_ema,
                        consecutive=consecutive_spikes,
                    )
                else:
                    consecutive_spikes = 0
                if resume_check is not None:
                    resume_check.append(mean_loss)
                    if len(resume_check) == 10:
                        try:
                            assert_resume_loss(
                                resume_check,
                                float(resolved.manifest.loss_ema) if resolved else 0.0,
                                float(cfg["resilience"]["resume_loss_tolerance"]),
                            )
                        except ResumeDiverged as exc:
                            state.status = "fatal"
                            state.event("resume_diverged", message=str(exc))
                            save_run_state(run_dir, state)
                            raise FatalResume(str(exc)) from exc
            else:
                scaler.update()
                opt.zero_grad(set_to_none=True)
                state.global_step += 1
                state.total_steps += 1
                state.skipped_steps += 1
                committed = cursor
                state.cursor = committed.to_dict()
                state.consumed_tokens = committed.consumed_tokens

            if finite and scale_before > scaler.get_scale():
                state.skipped_steps += 1
            step_elapsed = max(1e-6, time.monotonic() - step_start)
            tok_s = tokens_this_step / step_elapsed
            vram = torch.cuda.max_memory_allocated() / 2**30 if on_cuda else 0.0
            record = {
                "step": state.global_step,
                "global_step": state.global_step,
                "phase": phase,
                "loss": float(np.mean(step_losses)) if step_losses else None,
                "loss_ema": float(loss_ema),
                "lr": float(opt.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm) if finite else None,
                "tok_s": round(tok_s, 2),
                "vram_peak_gib": round(vram, 3),
                "skipped_steps": state.skipped_steps,
                "session_id": session_id,
                "elapsed": round(time.monotonic() - start_mono, 2),
                "consumed_tokens": state.consumed_tokens,
                "tokens": tokens_this_step,
                "optimizer_reset": state.optimizer_reset,
                "lr_scale": state.lr_scale,
                "oom_stage": state.oom_stage,
            }
            for k, values in step_extra.items():
                record[k] = round(float(np.mean(values)), 6)
            _meter(metrics_path, record)

            if consecutive_spikes >= 20:
                trigger_rollback("loss > 1.5x EMA for 20 consecutive steps")

            now = time.monotonic()
            since_ckpt = now - last_ckpt_time
            full_due = since_ckpt >= push.cadence.full_seconds
            weights_due = (
                push.cadence.weights_only_seconds > 0
                and now - last_ckpt_time >= push.cadence.weights_only_seconds
                and now - last_full_time < push.cadence.full_seconds
            )
            steps_due = state.global_step - last_ckpt_step >= ckpt_steps
            minutes_due = since_ckpt >= ckpt_minutes * 60
            if full_due or minutes_due or steps_due:
                checkpoint("full")
            elif weights_due:
                checkpoint("weights_only")

            if now - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = now
                payload = {
                    "run_id": run_id,
                    "phase": phase,
                    "step": state.global_step,
                    "loss_ema": round(float(loss_ema), 4),
                    "tok_s": round(tok_s, 2),
                    "vram_gib": round(vram, 2),
                    "session_age_s": int(time.time() - session_start),
                    "last_push_utc": push.last_success_utc,
                    "consumed_tokens": state.consumed_tokens,
                    "events": state.events[-3:],
                    "status": "training",
                    "utc": utcnow(),
                }
                try:
                    push.write_heartbeat(payload)
                except Exception as exc:
                    log(f"heartbeat failed (non-fatal): {exc}")

            if state.global_step % 10 == 0:
                save_run_state(run_dir, state)
            if state.global_step >= max_steps:
                status = "done"
                break

        if status == "done":
            state.status = "done"
        elif status == "data_exhausted":
            state.status = "data_exhausted"
        else:
            state.status = "stopped"
        try:
            if resolved is not None or state.global_step > 0:
                if state.global_step > last_ckpt_step or (
                    state.global_step == last_ckpt_step
                    and not (run_dir / f"ckpt-{state.global_step:08d}").exists()
                ):
                    checkpoint("full", force=True)
                push.wait()
        except Exception as exc:
            state.status = "push_failed"
            state.event("final_push_failed", message=str(exc)[:300])
            save_run_state(run_dir, state)
            raise T4TrainError(f"final checkpoint push failed: {exc}") from exc
        save_run_state(run_dir, state)
        _meter(metrics_path, {"event": "exit", "status": state.status, "step": state.global_step}, fsync=True)
        log(f"exit status={state.status} step={state.global_step}")
        return exit_code
    except (OOMRefused, FatalResume, DivergenceStop):
        raise
    except KeyboardInterrupt:
        status = "interrupted"
        try:
            checkpoint("full", force=True)
            push.wait()
        finally:
            state.status = "stopped"
            save_run_state(run_dir, state)
        return 0


def compute_phase_loss(
    phase: str,
    model: Any,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    *,
    on_cuda: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if phase == "sft":
        from .sft import sft_loss

        return sft_loss(model, batch, cfg, on_cuda=on_cuda)
    from .pref import dpo_loss

    return dpo_loss(model, batch, cfg, on_cuda=on_cuda)
