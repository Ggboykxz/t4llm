"""GPU end-to-end SFT/DPO tests on the tiny fixture, including Hub resume."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from t4llm import loop
from t4llm.fixtures import make_tiny_model
from t4llm.hub import LocalBackend, PushManager
from t4llm.pref import train_dpo
from t4llm.sft import train_sft
from t4llm.state import atomic_write_text, load_run_state

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("T4LLM_HUB_ROOT", str(tmp_path / "hub"))
    monkeypatch.setenv("T4LLM_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("T4LLM_ALLOW_SYNTHETIC", "1")
    monkeypatch.setenv("T4LLM_ALLOW_SMALL_BATCH", "1")
    monkeypatch.setattr(loop, "repo_root", lambda: tmp_path)
    yield tmp_path
    torch.cuda.empty_cache()


def _config(
    root: Path,
    phase: str,
    train_file: Path,
    *,
    packing: bool = False,
    neftune: float = 0.0,
) -> Path:
    model_dir = make_tiny_model(root / "tiny-model", seq_len=32)
    cfg = {
        "run": {
            "base_slug": "tiny",
            "phase": phase,
            "hub_repo": None,
            "data_repo": None,
            "data_revision": None,
            "seed": 3407,
        },
        "model": {
            "base": str(model_dir),
            "load_in_4bit": False,
            "attn": "sdpa",
            "gradient_checkpointing": "unsloth",
            "trust_remote_code": False,
        },
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "use_rslora": True,
            "target": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "train": {
            "dtype": "fp16",
            "seq_len": 32,
            "micro_batch": 1,
            "grad_accum": 1,
            "lr": 1e-3,
            "scheduler": "cosine",
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "optim": "paged_adamw_8bit",
            "max_steps": 4,
            "logging_steps": 1,
            "packing": packing,
            "num_train_epochs": 1,
        },
        "resilience": {
            "ckpt_every_minutes": 6,
            "ckpt_every_steps": 1,
            "heartbeat_seconds": 120,
            "time_budget_minutes": 5,
            "keep_last": 5,
            "keep_milestones_every": 1000,
            "push_backpressure_minutes": 15,
            "resume_loss_tolerance": 0.05,
            "max_rollbacks": 3,
        },
        "data": {
            "revision": None,
            "seq_len": 32,
            "local_dir": None,
            "source_manifest": None,
            "max_tokens_per_shard": 5_000_000,
        },
        "eval": {"domain_set": None, "general_samples": 32, "seed": 3407},
    }
    if phase == "sft":
        cfg["data"]["sft_file"] = str(train_file)
        cfg["train"]["train_on_responses_only"] = True
        cfg["train"]["neftune_noise_alpha"] = float(neftune)
    else:
        cfg["data"]["pref_file"] = str(train_file)
        cfg["train"]["beta"] = 0.1
        cfg["train"]["loss_type"] = "sigmoid"
    path = root / f"{phase}-config.json"
    atomic_write_text(path, json.dumps(cfg, indent=2) + "\n")
    return path


def _sft_rows(n: int = 8) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "prompt": f"tok{5 + i} tok{6 + i}",
                "response": f"tok{20 + i} tok{21 + i} tok{22 + i}",
                "source": "unit",
                "license": "CC0-1.0",
            }
        )
    return rows


def _dpo_rows(n: int = 8) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "prompt": f"tok{5 + i} tok{6 + i}",
                "chosen": f"tok{20 + i} tok{21 + i}",
                "rejected": f"tok{40 + i} tok{41 + i}",
                "source": "unit",
                "license": "CC0-1.0",
            }
        )
    return rows


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _metric_rows(root: Path, run_id: str) -> list[dict]:
    p = root / "runs" / run_id / "metrics.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_sft_trains_checkpoints_and_resumes(isolated):
    data = _write(isolated / "sft.jsonl", _sft_rows())
    cfg = _config(isolated / "sft", "sft", data)
    assert train_sft(cfg, time_budget="120s", max_steps=2, train_file=data) == 0

    mgr = PushManager(LocalBackend(isolated / "hub"))
    assert mgr.resolve_latest() == 2
    first = mgr.read_manifest(2)
    assert first.cursor["sample_idx"] == 2
    assert first.cursor["shard_idx"] == 0
    assert first.consumed_tokens > 0
    assert first.cursor["data_revision"].startswith("file-sha256:")

    rows = [
        r
        for r in _metric_rows(isolated, first.run_id)
        if "step" in r and "event" not in r
    ]
    assert [r["step"] for r in rows] == [1, 2]
    assert all(r["loss"] is not None and torch.isfinite(torch.tensor(r["loss"])) for r in rows)

    assert train_sft(cfg, time_budget="120s", max_steps=4, train_file=data) == 0
    second = mgr.read_manifest(4)
    assert second.run_id == first.run_id
    assert second.cursor["sample_idx"] == 4
    events = [
        json.loads(line)
        for line in (isolated / "runs" / first.run_id / "resume_events.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert events[-1]["from_step"] == 2
    assert events[-1]["lost_steps"] == 0
    state = load_run_state(isolated / "runs" / first.run_id)
    assert state is not None and state.global_step == 4


def test_dpo_trains_with_frozen_reference_and_resumes(isolated):
    data = _write(isolated / "dpo.jsonl", _dpo_rows())
    cfg = _config(isolated / "dpo", "dpo", data)
    assert train_dpo(cfg, time_budget="120s", max_steps=2, train_file=data) == 0

    mgr = PushManager(LocalBackend(isolated / "hub"))
    assert mgr.resolve_latest() == 2
    first = mgr.read_manifest(2)
    assert first.cursor["sample_idx"] == 2
    assert first.consumed_tokens > 0

    rows = [
        r
        for r in _metric_rows(isolated, first.run_id)
        if "step" in r and "event" not in r
    ]
    assert [r["step"] for r in rows] == [1, 2]
    for row in rows:
        assert row["loss"] is not None and torch.isfinite(torch.tensor(row["loss"]))
        assert "reward_margin" in row and torch.isfinite(torch.tensor(row["reward_margin"]))
        assert "dpo_accuracy" in row
        assert 0.0 <= row["dpo_accuracy"] <= 1.0

    assert train_dpo(cfg, time_budget="120s", max_steps=4, train_file=data) == 0
    second = mgr.read_manifest(4)
    assert second.run_id == first.run_id
    assert second.cursor["sample_idx"] == 4


def test_sft_refuses_packing(isolated):
    data = _write(isolated / "sft.jsonl", _sft_rows())
    cfg = _config(isolated / "sft-pack", "sft", data, packing=True)
    with pytest.raises(Exception, match="pack"):
        train_sft(cfg, time_budget="60s", max_steps=1, train_file=data)


def test_sft_neftune_trains_with_gradient_checkpointing(isolated):
    """NEFTune's inputs_embeds path must survive checkpointing + PEFT."""
    data = _write(isolated / "sft.jsonl", _sft_rows())
    cfg = _config(isolated / "sft-neftune", "sft", data, neftune=5.0)
    assert train_sft(cfg, time_budget="120s", max_steps=2, train_file=data) == 0

    mgr = PushManager(LocalBackend(isolated / "hub"))
    manifest = mgr.read_manifest(2)
    rows = [
        r
        for r in _metric_rows(isolated, manifest.run_id)
        if "step" in r and "event" not in r
    ]
    assert [r["step"] for r in rows] == [1, 2]
    assert all(r["loss"] is not None and torch.isfinite(torch.tensor(r["loss"])) for r in rows)
    # The run identity must include the NEFTune alpha; otherwise two different
    # optimisation trajectories could share a run_id.
    plain = _config(isolated / "sft-plain", "sft", data)
    from t4llm.phase_core import load_phase_config

    noisy_prepared = load_phase_config(cfg, train_file=data, require_t4=False)
    plain_prepared = load_phase_config(plain, train_file=data, require_t4=False)
    assert noisy_prepared.cfg_hash != plain_prepared.cfg_hash


@pytest.mark.parametrize("phase", ["sft", "dpo"])
def test_supervise_dispatches_phase_trainers(isolated, phase):
    """The real notebook entry point is `supervise`, not the direct trainer."""
    from t4llm.loop import supervise

    rows = _sft_rows() if phase == "sft" else _dpo_rows()
    data = _write(isolated / f"sup-{phase}.jsonl", rows)
    cfg = _config(isolated / f"sup-{phase}", phase, data)

    assert supervise(
        cfg, time_budget="180s", max_steps=1, train_file=data
    ) == 0

    mgr = PushManager(LocalBackend(isolated / "hub"))
    assert mgr.resolve_latest() == 1
    manifest = mgr.read_manifest(1)
    assert manifest.run_id.startswith(f"tiny-{phase}-")
    assert manifest.cursor["data_revision"].startswith("file-sha256:")
    assert manifest.cursor["sample_idx"] == 1
    state = load_run_state(isolated / "runs" / manifest.run_id)
    assert state is not None and state.global_step == 1
    assert state.status == "done"
