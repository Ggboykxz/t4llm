"""Resume contract tests (§9): fallback, config guard, RNG/optimizer restoration."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from t4llm.hub import LocalBackend, PushManager
from t4llm.resume import (
    ResumeDiverged,
    append_resume_event,
    assert_resume_loss,
    load_optimizer,
    load_rng_state,
    load_scheduler,
    read_resume_events,
    resolve_and_download,
    save_optimizer,
    save_rng_state,
    save_scheduler,
)
from t4llm.state import Cursor, RunState, build_manifest, save_run_state

from helpers import make_fake_checkpoint


def test_weights_only_checkpoint_requires_optimizer_reset(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    ckpt = make_fake_checkpoint(tmp_path / "local", step=8, tier="weights_only")
    PushManager(backend).publish(ckpt, step=8)
    resolved = resolve_and_download(
        backend, cache_dir=tmp_path / "cache", expected_config_hash="a1b2c3d4"
    )
    assert resolved is not None
    assert resolved.optimizer_reset is True
    assert resolved.manifest.tier == "weights_only"


def test_malformed_manifest_falls_back(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    for step in (5, 9):
        ckpt = make_fake_checkpoint(tmp_path / f"l{step}", step=step)
        PushManager(backend).publish(ckpt, step=step)
    (tmp_path / "hub" / "ckpt-00000009" / "manifest.json").write_text("{not json")
    resolved = resolve_and_download(
        backend, cache_dir=tmp_path / "cache", expected_config_hash="a1b2c3d4"
    )
    assert resolved is not None
    assert resolved.manifest.global_step == 5
    assert 9 in resolved.corrupt_steps


def test_no_valid_checkpoint_reports_candidates(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    ckpt = make_fake_checkpoint(tmp_path / "local", step=5)
    PushManager(backend).publish(ckpt, step=5)
    (tmp_path / "hub" / "ckpt-00000005" / "adapter_model.safetensors").write_bytes(b"bad")
    with pytest.raises(Exception, match="no valid checkpoint"):
        resolve_and_download(
            backend, cache_dir=tmp_path / "cache", expected_config_hash="a1b2c3d4"
        )


def test_rng_state_roundtrip_includes_all_generators(tmp_path):
    save_rng_state(tmp_path / "rng.pt")
    py_expected = random.random()
    np_expected = float(np.random.rand())
    torch_expected = torch.rand(1).item()
    # Advance every generator, then restore.
    random.random()
    np.random.rand(3)
    torch.rand(3)
    load_rng_state(tmp_path / "rng.pt")
    assert random.random() == py_expected
    assert float(np.random.rand()) == np_expected
    assert torch.rand(1).item() == torch_expected


def test_optimizer_and_scheduler_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 ** s)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    sched.step()
    save_optimizer(opt, tmp_path / "optimizer.pt")
    save_scheduler(sched, tmp_path / "scheduler.pt")

    model2 = torch.nn.Linear(4, 3)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 0.5 ** s)
    load_optimizer(opt2, tmp_path / "optimizer.pt")
    load_scheduler(sched2, tmp_path / "scheduler.pt")
    assert opt2.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    for a, b in zip(opt.state.values(), opt2.state.values()):
        for k in a:
            if torch.is_tensor(a[k]):
                assert torch.equal(a[k], b[k])


def test_resume_loss_guard():
    assert assert_resume_loss([1.0] * 10, 1.0) == 1.0
    assert_resume_loss([1.04] * 10, 1.0)  # inside 5%
    with pytest.raises(ResumeDiverged):
        assert_resume_loss([2.0] * 10, 1.0)
    with pytest.raises(ResumeDiverged, match="need 10"):
        assert_resume_loss([1.0] * 9, 1.0)


def test_resume_events_are_append_only(tmp_path):
    p = tmp_path / "resume_events.jsonl"
    append_resume_event(p, {"event": "resumed", "from_step": 10})
    append_resume_event(p, {"event": "resumed", "from_step": 20})
    events = read_resume_events(p)
    assert [e["from_step"] for e in events] == [10, 20]
    assert all("utc" in e for e in events)


def test_run_state_cursor_must_match_manifest(tmp_path):
    d = make_fake_checkpoint(tmp_path / "ckpt", step=10)
    raw = json.loads((d / "manifest.json").read_text())
    raw["consumed_tokens"] += 1
    (d / "manifest.json").write_text(json.dumps(raw))
    from t4llm.state import ManifestError, read_manifest

    with pytest.raises(ManifestError, match="consumed_tokens disagrees"):
        read_manifest(d)
