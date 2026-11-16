"""GPU end-to-end tests on the tiny fixture (fast, but exercise real fp16)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from t4llm import loop
from t4llm.fixtures import make_tiny_run
from t4llm.hub import PushManager, LocalBackend
from t4llm.state import read_manifest

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    monkeypatch.setenv("T4LLM_HUB_ROOT", str(hub))
    monkeypatch.setenv("T4LLM_ALLOW_SYNTHETIC", "1")
    monkeypatch.setenv("T4LLM_ALLOW_SMALL_BATCH", "1")
    monkeypatch.setattr(loop, "repo_root", lambda: tmp_path)
    yield tmp_path
    torch.cuda.empty_cache()


def _run(cfg_path, data_dir, steps):
    return loop.train(
        cfg_path,
        time_budget="120s",
        max_steps=steps,
        allow_synthetic=True,
        data_dir=data_dir,
        device="cuda",
    )


def test_training_resumes_from_hub_checkpoint(isolated):
    meta = make_tiny_run(
        isolated / "fixture",
        seq_len=64,
        num_samples=400,
        max_steps=6,
        ckpt_steps=2,
    )
    assert _run(meta["config"], meta["data"], 6) == 0
    backend = LocalBackend(isolated / "hub")
    mgr = PushManager(backend)
    assert mgr.resolve_latest() == 6
    first = mgr.read_manifest(6)
    assert first.global_step == 6
    assert first.cursor["consumed_tokens"] == 6 * 64

    assert _run(meta["config"], meta["data"], 12) == 0
    assert mgr.resolve_latest() == 12
    second = mgr.read_manifest(12)
    assert second.run_id == first.run_id
    assert second.global_step == 12
    assert second.cursor["consumed_tokens"] == 12 * 64
    assert second.cursor["sample_idx"] == 12
    metrics = [
        json.loads(line)
        for line in (isolated / "runs" / first.run_id / "metrics.jsonl").read_text().splitlines()
    ]
    assert any(m.get("event") == "resume" and m["resumed_from_step"] == 6 for m in metrics)
    events = [
        json.loads(line)
        for line in (isolated / "runs" / first.run_id / "resume_events.jsonl").read_text().splitlines()
    ]
    assert events[0]["from_step"] == 6
    assert events[0]["lost_steps"] == 0


def test_batch_crosses_shard_boundary_without_losing_samples(isolated):
    meta = make_tiny_run(
        isolated / "fixture2",
        seq_len=16,
        num_samples=60,
        max_steps=3,
        ckpt_steps=1,
        micro_batch=10,
        grad_accum=1,
        max_samples_per_shard=25,
    )
    assert _run(meta["config"], meta["data"], 3) == 0
    mgr = PushManager(LocalBackend(isolated / "hub"))
    m = mgr.read_manifest(3)
    # Batch 3 takes 5 samples from shard 0 and 5 from shard 1; the cursor is
    # the exact post-batch position, not a per-shard approximation.
    assert m.cursor["shard_idx"] == 1
    assert m.cursor["sample_idx"] == 5
    assert m.cursor["consumed_tokens"] == 30 * 16
    run_dir = isolated / "runs" / m.run_id
    lines = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if '"step":' in line and '"event"' not in line
    ]
    assert [line["consumed_tokens"] for line in lines[:3]] == [160, 320, 480]


def test_eval_and_export_on_tiny_run(isolated):
    from t4llm.eval import run_eval
    from t4llm.export import export_model

    meta = make_tiny_run(
        isolated / "fixture3", seq_len=32, num_samples=200, max_steps=2, ckpt_steps=1
    )
    assert _run(meta["config"], meta["data"], 2) == 0
    domain = isolated / "domain.jsonl"
    rows = [
        {"text": "tok5 tok6 tok7 tok8 tok9 tok10 tok11 tok12"},
        {"text": "tok25 tok26 tok27 tok28 tok29 tok30 tok31 tok32"},
        {"text": "tok45 tok46 tok47 tok48 tok49 tok50 tok51 tok52"},
    ]
    domain.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")
    import json as _json

    cfg = _json.loads(Path(meta["config"]).read_text())
    cfg["eval"]["domain_set"] = str(domain)
    cfg_path = isolated / "eval-config.json"
    cfg_path.write_text(_json.dumps(cfg, indent=2))

    result = run_eval(str(cfg_path), limit=3)
    assert result["domain_ppl"] > 0
    assert result["base_domain_ppl"] > 0
    assert result["general_ppl"] is None
    assert result["regression"] is False
    assert (isolated / "docs" / "EVAL.md").is_file()

    exported = export_model(str(cfg_path), gguf=False, push=True)
    assert Path(exported["gguf"]) if exported["gguf"] else True
    merged = list((isolated / "exports").rglob("model.safetensors")) + list(
        (isolated / "exports").rglob("model-*.safetensors")
    )
    assert merged, "merged safetensors not written"
    backend = LocalBackend(isolated / "hub")
    prefix = backend.read_bytes("EXPORT_LATEST").decode().strip()
    assert prefix == "exports/step-00000002"
    assert backend.list_files(prefix + "/model")
    remote_manifest = _json.loads(backend.read_bytes(prefix + "/export_manifest.json"))
    assert remote_manifest["merged_dtype"] == "fp16"
    assert "model.safetensors" in remote_manifest["files"]
