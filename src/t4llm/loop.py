"""Time-budgeted CPT loop, checkpointing and the supervising parent (§10).

The child process owns the model and is allowed to die.  The parent only reads
state and relaunches.  The wall-clock timer is the guarantee; signal handlers
are best effort.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import env
from .data import (
    DataError,
    DataExhausted,
    PackedCursor,
    fetch_data,
    load_index,
)
from .hub import (
    BackpressureError,
    FatalHubError,
    HubError,
    PushManager,
    make_backend,
)
from .modeling import (
    assert_no_bf16,
    build_model,
    build_optimizer,
    build_scheduler,
    configure_cuda,
    load_adapter,
    save_adapter,
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
    save_optimizer,
    save_rng_state,
    save_scheduler,
)
from .state import (
    ConfigMismatchError,
    ConfigError,
    Cursor,
    RunState,
    approx_tokens_per_step,
    atomic_write_json,
    atomic_write_text,
    build_manifest,
    compute_config_hash,
    load_config,
    load_run_state,
    make_run_id,
    parse_duration,
    save_run_state,
    utcnow,
    validate_checkpoint_dir,
    write_trainer_state,
)


class T4TrainError(RuntimeError):
    exit_code = 2


class OOMRefused(T4TrainError):
    exit_code = 3


class DivergenceStop(T4TrainError):
    exit_code = 4


class FatalResume(T4TrainError):
    exit_code = 5


def log(msg: str) -> None:
    print(f"[t4llm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    """Run artifacts live here.  Overridable for chaos/CI isolation."""
    explicit = os.environ.get("T4LLM_RUNS_ROOT")
    return Path(explicit) if explicit else repo_root() / "runs"


def run_dir_for(run_id: str) -> Path:
    return runs_root() / run_id


@dataclass
class Prepared:
    cfg: dict[str, Any]
    index: dict[str, Any]
    data_root: Path
    cfg_hash: str
    run_id: str


def prepare_run(
    config_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    model: str | None = None,
    run_id: str | None = None,
    allow_synthetic: bool = False,
) -> Prepared:
    overrides: dict[str, Any] = {}
    if data_dir:
        overrides.setdefault("data", {})["local_dir"] = str(data_dir)
    if model:
        overrides.setdefault("model", {})["base"] = model
    cfg = load_config(config_path, overrides)
    if str(cfg["run"]["phase"]) != "cpt":
        raise ConfigError(
            f"phase {cfg['run']['phase']!r} is not the CPT token-packing loop; "
            "use the phase-specific trainer (sft/dpo) so prompt masking and the "
            "reference model are handled correctly"
        )
    from .state import _params_b

    env.require_t4(
        dtype=cfg["train"]["dtype"],
        attn=cfg["model"]["attn"],
        load_in_4bit=bool(cfg["model"].get("load_in_4bit")),
        model_params_b=_params_b(cfg["model"]["base"]) or None,
    )
    local = cfg["data"].get("local_dir")
    seq_len = int(cfg["train"]["seq_len"])
    if local:
        index = load_index(local)
        data_root = Path(local)
    else:
        rev = cfg["run"].get("data_revision") or cfg["data"].get("revision")
        if not rev:
            raise ConfigError(
                "no pinned data revision: run `make data`, set run.data_revision, "
                "or pass --data-dir for a test transport"
            )
        backend = make_backend(cfg, "dataset")
        index, data_root = fetch_data(
            backend, str(rev), runs_root() / "_data-cache", expected_seq_len=seq_len
        )
    if int(index["seq_len"]) != seq_len:
        raise DataError(f"data seq_len {index['seq_len']} != train seq_len {seq_len}")
    if index.get("synthetic") and not allow_synthetic and os.environ.get("T4LLM_ALLOW_SYNTHETIC") != "1":
        raise DataError(
            "refusing to train on synthetic test shards; pass --allow-synthetic only for chaos/CI"
        )
    # Freeze the data revision into the run identity if the operator left it open.
    if not cfg["run"].get("data_revision"):
        cfg["run"]["data_revision"] = index["data_revision"]
        cfg["data"]["revision"] = index["data_revision"]
    elif cfg["run"]["data_revision"] != index["data_revision"]:
        raise ConfigError(
            f"configured data revision {cfg['run']['data_revision']} != loaded "
            f"{index['data_revision']}; refusing to train on moved data"
        )
    h = compute_config_hash(cfg)
    rid = run_id or make_run_id(cfg, h)
    return Prepared(cfg=cfg, index=index, data_root=data_root, cfg_hash=h, run_id=rid)


def _acquire_lock(run_dir: Path):
    import fcntl

    run_dir.mkdir(parents=True, exist_ok=True)
    fh = open(run_dir / "train.lock", "w", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise T4TrainError(f"another trainer already holds {run_dir}/train.lock") from exc
    fh.write(f"pid={os.getpid()} started={utcnow()}\n")
    fh.flush()
    return fh


def _meter(
    path: Path,
    record: dict[str, Any],
    *,
    fsync: bool = False,
) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())


def _load_tokenizer(cfg: dict[str, Any]):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(
            cfg["model"]["base"], trust_remote_code=bool(cfg["model"].get("trust_remote_code"))
        )
    except Exception as exc:
        raise T4TrainError(f"tokenizer load failed for {cfg['model']['base']}: {exc}") from exc
    return tok


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _save_checkpoint(
    *,
    run_dir: Path,
    step: int,
    tier: str,
    state: RunState,
    cfg: dict[str, Any],
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cursor: Cursor,
    loss_ema: float,
    forked: bool,
    optimizer_reset: bool,
    runtime_notes: list[str],
    env_report: dict[str, Any],
    tokenizer: Any,
) -> Path:
    final = run_dir / f"ckpt-{step:08d}"
    if final.exists():
        try:
            validate_checkpoint_dir(final)
            return final
        except Exception as exc:
            raise T4TrainError(
                f"local checkpoint {final} exists but is invalid ({exc}); refusing to overwrite"
            ) from exc
    tmp = run_dir / f".ckpt-{step:08d}-{uuid.uuid4().hex[:8]}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        save_adapter(model, tmp)
        if tier == "full":
            save_optimizer(optimizer, tmp / "optimizer.pt")
            save_scheduler(scheduler, tmp / "scheduler.pt")
            save_rng_state(tmp / "rng_state.pt")
        state.global_step = step
        state.cursor = cursor.to_dict()
        state.consumed_tokens = cursor.consumed_tokens
        state.loss_ema = float(loss_ema)
        state.last_ckpt_step = step
        write_trainer_state(
            tmp / "trainer_state.json",
            state=state,
            config=cfg,
            extra={"runtime_notes": runtime_notes, "tier": tier},
        )
        tokenizer.save_pretrained(str(tmp / "tokenizer"))
        atomic_write_json(tmp / "config.json", cfg)
        atomic_write_json(tmp / "env.json", env_report)
        build_manifest(
            tmp,
            run_id=state.run_id,
            global_step=step,
            tier=tier,
            consumed_tokens=cursor.consumed_tokens,
            cursor=cursor,
            config_hash=state.config_hash,
            loss_ema=float(loss_ema),
            forked=forked,
            optimizer_reset=optimizer_reset,
        )
        for p in tmp.rglob("*"):
            if p.is_file():
                try:
                    with open(p, "rb") as fh:
                        os.fsync(fh.fileno())
                except OSError:
                    pass
        _fsync_dir(tmp)
        os.replace(tmp, final)
        _fsync_dir(run_dir)
        log(f"checkpoint {final.name} ({tier}) written atomically")
        return final
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _scaled_step(
    scheduler: Any,
    optimizer: torch.optim.Optimizer,
    *,
    lr_scale: float,
    reset_warmup_remaining: int,
    reset_warmup_total: int = 20,
) -> None:
    scheduler.step()
    factor = float(lr_scale)
    if reset_warmup_remaining > 0:
        done = reset_warmup_total - reset_warmup_remaining + 1
        factor *= min(1.0, done / reset_warmup_total)
    if factor != 1.0:
        for group in optimizer.param_groups:
            group["lr"] *= factor


def _apply_lr_scale(optimizer: torch.optim.Optimizer, scale: float) -> None:
    if scale != 1.0:
        for group in optimizer.param_groups:
            group["lr"] *= scale


def _read_rollback_marker(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def train(
    config_path: str | Path,
    *,
    time_budget: str | int | float | None = None,
    max_steps: int | None = None,
    run_id: str | None = None,
    allow_config_change: bool = False,
    allow_synthetic: bool = False,
    data_dir: str | Path | None = None,
    model: str | None = None,
    device: str | None = None,
) -> int:
    configure_cuda()
    prepared = prepare_run(
        config_path,
        data_dir=data_dir,
        model=model,
        run_id=run_id,
        allow_synthetic=allow_synthetic,
    )
    cfg, index, data_root = prepared.cfg, prepared.index, prepared.data_root
    rid, cfg_hash = prepared.run_id, prepared.cfg_hash
    run_dir = run_dir_for(rid)
    lock = _acquire_lock(run_dir)
    try:
        return _train_inner(
            cfg=cfg,
            index=index,
            data_root=data_root,
            run_id=rid,
            cfg_hash=cfg_hash,
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


def _train_inner(
    *,
    cfg: dict[str, Any],
    index: dict[str, Any],
    data_root: Path,
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
    # Contract: checkpoint at budget - 90 s.  Short CI budgets use 60%.
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
        cursor_state = resolved.manifest.cursor
        loss_ema = float(resolved.manifest.loss_ema)
        optimizer_reset = resolved.optimizer_reset or resolved.forked
        forked = resolved.forked
        lost_steps = max(0, (previous_state.global_step if previous_state else start_step) - start_step)
        state = previous_state or RunState(
            run_id=run_id, phase=cfg["run"]["phase"], config_hash=cfg_hash
        )
        state.global_step = start_step
        state.cursor = dict(cursor_state)
        state.consumed_tokens = int(cursor_state["consumed_tokens"])
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
            },
        )
        for bad in resolved.corrupt_steps:
            state.event("corrupt_checkpoint", step=bad)
        log(
            f"resuming {run_id} at step {start_step} (lost {lost_steps}, "
            f"optimizer_reset={optimizer_reset}, corrupt={resolved.corrupt_steps})"
        )
    else:
        start_step = 0
        loss_ema = 0.0
        optimizer_reset = False
        forked = False
        lost_steps = 0
        state = RunState(run_id=run_id, phase=cfg["run"]["phase"], config_hash=cfg_hash)
        state.cursor = Cursor(
            data_revision=index["data_revision"],
            shard_idx=0,
            sample_idx=0,
            consumed_tokens=0,
        ).to_dict()
        state.event("started", max_steps=max_steps, data_revision=index["data_revision"])
        log(f"starting fresh run {run_id}")

    state.session_id = session_id
    save_run_state(run_dir, state)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        # A new session may have no local metrics; recover the Hub copy if present.
        try:
            metrics_path.write_bytes(backend.read_bytes("metrics.jsonl"))
        except HubError:
            metrics_path.touch()

    # ----- data cursor ----------------------------------------------------- #
    verify_hashes = os.environ.get("T4LLM_VERIFY_DATA_HASHES") == "1"
    cursor = PackedCursor(index, data_root, verify_hashes=verify_hashes)
    if resolved is not None:
        c = resolved.manifest.cursor
        cursor.seek(int(c["shard_idx"]), int(c["sample_idx"]), int(c["consumed_tokens"]))
    committed = cursor.cursor()

    # ----- model ----------------------------------------------------------- #
    set_seed(int(cfg["run"]["seed"]) + (start_step if optimizer_reset else 0))
    log(f"building {cfg['model']['base']} (fp16 LoRA, {cfg['model']['attn']})")
    bundle = build_model(cfg, device=device, attach_lora=resolved is None)
    model = bundle.model
    runtime_notes = list(bundle.notes)
    if resolved is not None:
        model = load_adapter(model, resolved.local_dir)
    if torch.cuda.is_available() and (device is None or device.startswith("cuda")):
        from .modeling import cast_trainable_fp32

        cast_trainable_fp32(model)
    assert_no_bf16(model)
    on_cuda = torch.cuda.is_available() and (device is None or device.startswith("cuda"))
    tokenizer = _load_tokenizer(cfg)
    push.mark_started()

    opt, opt_notes = build_optimizer(model, cfg, allow_fallback=not on_cuda)
    runtime_notes.extend(opt_notes)
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

    # Divergence marker: the supervisor relaunches after a stop; the marker is
    # consumed exactly once and persisted in state.lr_scale.
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
    step_times: list[float] = []
    resume_check: list[float] = (
        [] if (resolved is not None and not forked and not optimizer_reset) else None
    )
    skip_ratio_alerted = False
    consecutive_spikes = 0
    env_report = env.capability_report()
    session_start = time.time()

    def trigger_rollback(reason: str) -> None:
        if state.rollbacks + 1 > int(cfg["resilience"]["max_rollbacks"]):
            state.status = "fatal"
            state.event("divergence_halt", rollbacks=state.rollbacks + 1, reason=reason)
            save_run_state(run_dir, state)
            raise DivergenceStop(f"{reason}: divergence rollback limit reached")
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
        f"run={run_id} step={start_step} max_steps={max_steps} "
        f"budget={budget_seconds}s effective_tokens={effective_tokens} "
        f"data={index['data_revision'][:12]}"
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
            # Backpressure is checked at every step, not only at checkpoints.
            try:
                push.check_backpressure()
            except BackpressureError as exc:
                log(f"backpressure: {exc}; waiting for the in-flight push")
                push.wait()
                push.check_backpressure()

            opt.zero_grad(set_to_none=True)
            accumulated = 0
            step_losses: list[float] = []
            step_start = time.monotonic()
            tokens_this_step = 0
            try:
                while accumulated < grad_accum:
                    if time.monotonic() >= hard_deadline:
                        break
                    try:
                        batch_np, batch_cursor = cursor.next_batch(micro_batch)
                    except DataExhausted:
                        status = "data_exhausted"
                        log("data exhausted; stopping rather than repeating silently")
                        break
                    input_ids = torch.from_numpy(batch_np).long()
                    if on_cuda:
                        input_ids = input_ids.to("cuda", non_blocking=True)
                    with torch.autocast(
                        device_type="cuda" if on_cuda else "cpu",
                        dtype=torch.float16,
                        enabled=on_cuda,
                    ):
                        out = model(input_ids=input_ids, labels=input_ids)
                        loss = out.loss
                    if loss is None or not torch.isfinite(loss):
                        cursor.seek(
                            committed.shard_idx,
                            committed.sample_idx,
                            committed.consumed_tokens,
                        )
                        trigger_rollback("non-finite loss")
                    scaler.scale(loss / grad_accum).backward()
                    accumulated += 1
                    tokens_this_step += int(batch_np.shape[0]) * seq_len
                    step_losses.append(float(loss.detach().float().cpu()))
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                cursor.seek(
                    committed.shard_idx, committed.sample_idx, committed.consumed_tokens
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
                # Budget hit inside an accumulation window: rewind and checkpoint
                # only the last fully committed optimiser step.
                cursor.seek(committed.shard_idx, committed.sample_idx, committed.consumed_tokens)
                break
            if accumulated < grad_accum:
                # Partial accumulation must not be committed; discard it.
                opt.zero_grad(set_to_none=True)
                cursor.seek(committed.shard_idx, committed.sample_idx, committed.consumed_tokens)
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
                committed = batch_cursor
                state.cursor = committed.to_dict()
                state.consumed_tokens = committed.consumed_tokens
                mean_loss = float(np.mean(step_losses))
                previous_ema = loss_ema
                if loss_ema <= 0:
                    loss_ema = mean_loss
                else:
                    loss_ema = 0.98 * loss_ema + 0.02 * mean_loss
                state.loss_ema = float(loss_ema)
                # §10: 20 consecutive steps above 1.5x the pre-step EMA.
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
                # The scaler found an inf; skip this optimiser step but consume
                # the same data as an uninterrupted run would.
                scaler.update()
                opt.zero_grad(set_to_none=True)
                state.global_step += 1
                state.total_steps += 1
                state.skipped_steps += 1
                committed = batch_cursor
                state.cursor = committed.to_dict()
                state.consumed_tokens = committed.consumed_tokens

            if finite and scale_before > scaler.get_scale():
                state.skipped_steps += 1
            step_elapsed = max(1e-6, time.monotonic() - step_start)
            step_times.append(step_elapsed)
            if len(step_times) > 50:
                step_times.pop(0)
            tok_s = tokens_this_step / step_elapsed
            vram = (
                torch.cuda.max_memory_allocated() / 2**30 if on_cuda else 0.0
            )
            record = {
                "step": state.global_step,
                "global_step": state.global_step,
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
            _meter(metrics_path, record)
            if state.total_steps > 20 and state.skipped_steps / max(1, state.total_steps) > 0.05:
                if not skip_ratio_alerted:
                    skip_ratio_alerted = True
                    state.event(
                        "high_skip_ratio",
                        skipped=state.skipped_steps,
                        total=state.total_steps,
                    )
                    log(
                        f"ALERT: skipped-step ratio {state.skipped_steps/state.total_steps:.1%} > 5%"
                    )

            # Divergence before checkpointing: never publish a spike checkpoint.
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
        # Force a full checkpoint at every clean exit, so a time-budget stop is
        # never worse than the normal cadence.
        try:
            if resolved is not None or state.global_step > 0:
                if state.global_step > last_ckpt_step or (
                    state.global_step == last_ckpt_step and not (run_dir / f"ckpt-{state.global_step:08d}").exists()
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


def supervise(
    config_path: str | Path,
    *,
    time_budget: str | int | float | None = None,
    max_steps: int | None = None,
    allow_config_change: bool = False,
    allow_synthetic: bool = False,
    data_dir: str | Path | None = None,
    model: str | None = None,
    train_file: str | Path | None = None,
) -> int:
    preview = load_config(config_path)
    phase = str(preview["run"]["phase"])
    if phase == "cpt":
        prepared = prepare_run(
            config_path,
            data_dir=data_dir,
            model=model,
            allow_synthetic=allow_synthetic,
        )
    else:
        from .phase_core import prepare_phase_run

        prepared = prepare_phase_run(
            config_path, train_file=train_file, model=model
        )
    cfg, rid = prepared.cfg, prepared.run_id
    run_dir = run_dir_for(rid)
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = (
        parse_duration(time_budget)
        if time_budget is not None
        else int(cfg["resilience"]["time_budget_minutes"]) * 60
    )
    max_steps = int(max_steps if max_steps is not None else cfg["train"]["max_steps"])
    child_pid_file = run_dir / "child.pid"
    failures = 0
    stopping = False
    deadline = time.monotonic() + budget

    def _signal(signum, frame):  # noqa: ARG001
        nonlocal stopping
        stopping = True
        log(f"supervisor received signal {signum}; stopping child")
        try:
            pid = int(child_pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old_handlers[sig] = signal.signal(sig, _signal)
        except Exception:
            pass

    cmd = [
        sys.executable,
        "-m",
        "t4llm",
        "train" if phase == "cpt" else phase,
        "--config",
        str(config_path),
        "--max-steps",
        str(max_steps),
        "--run-id",
        rid,
    ]
    if allow_config_change:
        cmd.append("--allow-config-change")
    if allow_synthetic and phase == "cpt":
        cmd.append("--allow-synthetic")
    if data_dir and phase == "cpt":
        cmd += ["--data-dir", str(data_dir)]
    if train_file and phase != "cpt":
        cmd += ["--train-file", str(train_file)]
    if model:
        cmd += ["--model", model]
    log(f"supervising {rid} with an overall {budget}s budget")
    try:
        while not stopping:
            remaining = deadline - time.monotonic()
            if remaining <= 30:
                log("supervisor budget exhausted; leaving the last checkpoint in place")
                return 0
            child_budget = int(remaining)
            child_cmd = cmd + ["--time-budget", f"{child_budget}s"]
            proc = subprocess.Popen(child_cmd, env=os.environ.copy())
            child_pid_file.write_text(str(proc.pid) + "\n")
            rc = proc.wait()
            try:
                child_pid_file.unlink()
            except OSError:
                pass
            st = load_run_state(run_dir)
            step = st.global_step if st else 0
            status = st.status if st else "unknown"
            log(f"child exited rc={rc} step={step} status={status}")
            if rc == 0:
                failures = 0
                if status in {"done", "data_exhausted"} or step >= max_steps:
                    return 0
                # The child used the remaining budget and checkpointed.  A
                # notebook cell should end here rather than silently starting
                # another full training window.
                return 0
            if rc == 4:
                failures = 0
                if st and st.rollbacks > int(cfg["resilience"]["max_rollbacks"]):
                    raise T4TrainError("divergence rollback limit reached")
                log("relaunching after divergence rollback")
                continue
            if rc in (3, 5):
                raise T4TrainError(f"fatal child exit rc={rc}; see run log")
            if stopping:
                break
            failures += 1
            if failures >= 5:
                raise T4TrainError(f"child failed {failures} times; refusing to loop forever")
            delay = min(60, 2**failures)
            delay = min(delay, max(1, int(deadline - time.monotonic()) - 30))
            log(f"relaunching in {delay}s after rc={rc} (failure {failures}/5)")
            time.sleep(delay)
    finally:
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        try:
            child_pid_file.unlink()
        except OSError:
            pass
    return 0
