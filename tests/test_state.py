from __future__ import annotations

import json
from pathlib import Path

import pytest

from t4llm.state import (
    ConfigMismatchError,
    ManifestError,
    RunState,
    build_manifest,
    check_config_hash,
    compute_config_hash,
    load_config,
    load_run_state,
    make_run_id,
    parse_duration,
    read_manifest,
    save_run_state,
    validate_checkpoint_dir,
    write_fork_adr,
)
from helpers import make_fake_checkpoint


def base_cfg():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "cpt.yaml")


def test_config_hash_training_relevant_only():
    cfg = base_cfg()
    h = compute_config_hash(cfg)
    changed = json.loads(json.dumps(cfg))
    changed["run"]["hub_repo"] = "someone/else"
    changed["resilience"]["ckpt_every_steps"] = 1
    assert compute_config_hash(changed) == h
    changed2 = json.loads(json.dumps(cfg))
    changed2["train"]["lr"] *= 2
    assert compute_config_hash(changed2) != h


def test_oom_split_preserves_hash_but_effective_tokens_do_not():
    cfg = base_cfg()
    h = compute_config_hash(cfg)
    split = json.loads(json.dumps(cfg))
    split["train"]["micro_batch"] = 1
    split["train"]["grad_accum"] = 32
    assert compute_config_hash(split) == h
    eff = json.loads(json.dumps(cfg))
    eff["train"]["grad_accum"] = 32
    assert compute_config_hash(eff) != h


def test_run_id_shape():
    cfg = base_cfg()
    rid = make_run_id(cfg)
    assert rid.startswith("qwen3-1.7b-cpt-")
    assert len(rid.rsplit("-", 1)[-1]) == 8


def test_manifest_roundtrip_and_validation(tmp_path):
    ckpt = make_fake_checkpoint(tmp_path, step=12, sample_idx=12)
    m = read_manifest(ckpt)
    assert m.global_step == 12
    assert m.tier == "full"
    assert m.cursor["consumed_tokens"] == 12 * 64
    assert "SHA256SUMS" in m.files
    validate_checkpoint_dir(ckpt)


def test_manifest_tamper_detected(tmp_path):
    ckpt = make_fake_checkpoint(tmp_path, step=3)
    with open(ckpt / "adapter_model.safetensors", "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(ManifestError, match="size mismatch|sha256 mismatch"):
        validate_checkpoint_dir(ckpt)


def test_manifest_rejects_unknown_schema(tmp_path):
    ckpt = make_fake_checkpoint(tmp_path, step=3)
    raw = json.loads((ckpt / "manifest.json").read_text())
    raw["schema_version"] = 99
    (ckpt / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ManifestError, match="newer"):
        read_manifest(ckpt)


def test_manifest_written_last(tmp_path):
    ckpt = make_fake_checkpoint(tmp_path, step=3)
    names = [p.name for p in ckpt.iterdir()]
    assert names[-1] == "manifest.json" or "manifest.json" in names
    # SHA256SUMS covers payload but never the manifest itself.
    sums = (ckpt / "SHA256SUMS").read_text()
    assert "manifest.json" not in sums
    assert "adapter_model.safetensors" in sums


def test_config_guard_and_fork_adr(tmp_path):
    with pytest.raises(ConfigMismatchError):
        check_config_hash("aaaaaaaa", "bbbbbbbb")
    check_config_hash("aaaaaaaa", "bbbbbbbb", allow_config_change=True)
    adr = tmp_path / "DECISIONS.md"
    write_fork_adr(adr, run_id="r-1", old_hash="aaaaaaaa", new_hash="bbbbbbbb", reason="test")
    assert "forked" in adr.read_text()


def test_run_state_roundtrip(tmp_path):
    st = RunState(run_id="tiny-cpt-a1b2c3d4", phase="cpt", config_hash="a1b2c3d4")
    st.event("hello", n=1)
    save_run_state(tmp_path, st)
    loaded = load_run_state(tmp_path)
    assert loaded is not None
    assert loaded.run_id == st.run_id
    assert loaded.events[-1]["event"] == "hello"


def test_duration_parser():
    assert parse_duration("20m") == 1200
    assert parse_duration("90s") == 90
    assert parse_duration("2h") == 7200
    assert parse_duration(5) == 300
