"""Hub contract tests (§8): atomic publish, verification, fallback, retention."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from tenacity import retry

from t4llm import hub
from t4llm.hub import (
    BackpressureError,
    CheckpointExistsError,
    CorruptCheckpointError,
    FatalHubError,
    LocalBackend,
    PushManager,
    probe_upload,
)
from t4llm.state import ConfigMismatchError, read_manifest
from t4llm.resume import resolve_and_download

from helpers import make_fake_checkpoint


def publish(backend, ckpt, step: int, **kwargs):
    return PushManager(backend, **kwargs).publish(ckpt, step=step)


def test_publish_is_visible_only_after_full_commit(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    ckpt = make_fake_checkpoint(tmp_path / "local", step=10)
    mgr = PushManager(backend)
    result = mgr.publish(ckpt, step=10)
    assert result["step"] == 10
    assert mgr.resolve_latest() == 10
    assert backend.read_bytes("LATEST").decode().strip() == "ckpt-00000010"
    remote_manifest = mgr.read_manifest(10)
    assert remote_manifest.run_id == read_manifest(ckpt).run_id
    assert backend.list_files("ckpt-00000010/")


def test_crash_mid_upload_leaves_no_partial_checkpoint(tmp_path):
    class DyingBackend(LocalBackend):
        def upload_folder(self, local_dir, dest_prefix, message):
            # Simulate kill -9 after the first file reached staging.
            staging = self._staging / "dead-session"
            staging.mkdir(parents=True, exist_ok=True)
            src = Path(local_dir)
            first = sorted(p for p in src.rglob("*") if p.is_file())[0]
            shutil.copyfile(first, staging / first.name)
            raise RuntimeError("simulated crash during upload")

    backend = DyingBackend(tmp_path / "hub")
    ckpt = make_fake_checkpoint(tmp_path / "local", step=10)
    with pytest.raises(RuntimeError, match="simulated crash"):
        publish(backend, ckpt, 10)
    assert backend.list_files("ckpt-") == []
    assert PushManager(backend).list_checkpoint_steps() == []
    with pytest.raises(FatalHubError):
        backend.read_bytes("LATEST")
    # The partial stage is invisible and can never be mistaken for a commit.
    assert list((tmp_path / "hub" / ".staging").iterdir())


def test_existing_checkpoint_is_immutable(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    ckpt = make_fake_checkpoint(tmp_path / "local", step=5)
    publish(backend, ckpt, 5)
    with pytest.raises(CheckpointExistsError):
        publish(backend, ckpt, 5)


def test_verify_remote_detects_payload_tamper(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    ckpt = make_fake_checkpoint(tmp_path / "local", step=7)
    mgr = PushManager(backend)
    mgr.publish(ckpt, step=7)
    p = tmp_path / "hub" / "ckpt-00000007" / "adapter_model.safetensors"
    with open(p, "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(CorruptCheckpointError):
        mgr.verify_remote(7, mgr.read_manifest(7))


def test_corrupt_latest_falls_back_to_previous(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    cfg_hash = "a1b2c3d4"
    for step in (10, 20):
        ckpt = make_fake_checkpoint(
            tmp_path / f"local{step}", step=step, config_hash=cfg_hash
        )
        publish(backend, ckpt, step)
    p = tmp_path / "hub" / "ckpt-00000020" / "adapter_model.safetensors"
    with open(p, "ab") as fh:
        fh.write(b"bitrot")
    resolved = resolve_and_download(
        backend,
        cache_dir=tmp_path / "cache",
        expected_config_hash=cfg_hash,
    )
    assert resolved is not None
    assert resolved.manifest.global_step == 10
    assert resolved.corrupt_steps == [20]
    assert not (tmp_path / "cache" / "ckpt-00000020").exists()


def test_missing_latest_file_falls_back_to_highest_checkpoint(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    ckpt = make_fake_checkpoint(tmp_path / "local", step=30)
    mgr = PushManager(backend)
    mgr.publish(ckpt, step=30)
    backend.upload_bytes(b"garbage\n", "LATEST", "test corruption")
    assert mgr.resolve_latest() == 30


def test_config_guard_blocks_and_fork_writes_adr(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    ckpt = make_fake_checkpoint(tmp_path / "local", step=4, config_hash="aaaaaaaa")
    publish(backend, ckpt, 4)
    with pytest.raises(ConfigMismatchError):
        resolve_and_download(
            backend,
            cache_dir=tmp_path / "cache",
            expected_config_hash="bbbbbbbb",
        )
    adr = tmp_path / "DECISIONS.md"
    resolved = resolve_and_download(
        backend,
        cache_dir=tmp_path / "cache2",
        expected_config_hash="bbbbbbbb",
        allow_config_change=True,
        adr_path=adr,
        reason="test fork",
    )
    assert resolved is not None
    assert resolved.forked is True
    assert resolved.optimizer_reset is True
    text = adr.read_text()
    assert "aaaaaaaa" in text and "bbbbbbbb" in text and "test fork" in text


def test_retention_keeps_latest_milestones_and_full_checkpoints(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    mgr = PushManager(backend, keep_last=3, keep_milestones_every=4)
    for step in range(1, 9):
        ckpt = make_fake_checkpoint(tmp_path / f"local{step}", step=step)
        mgr.publish(ckpt, step=step)
    steps = mgr.list_checkpoint_steps()
    # latest (8), last three (6,7,8), and milestones (4,8) are retained.
    assert steps == [4, 6, 7, 8]
    assert mgr.resolve_latest() == 8
    assert not (tmp_path / "hub" / "ckpt-00000001").exists()
    assert not (tmp_path / "hub" / "ckpt-00000005").exists()


def test_weights_only_never_evicts_all_full_checkpoints(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    mgr = PushManager(backend, keep_last=2, keep_milestones_every=1000)
    for step in (10, 20, 30, 40):
        tier = "full" if step in (10, 20) else "weights_only"
        ckpt = make_fake_checkpoint(tmp_path / f"local{step}", step=step, tier=tier)
        mgr.publish(ckpt, step=step)
    steps = mgr.list_checkpoint_steps()
    tiers = {s: mgr.read_manifest(s).tier for s in steps}
    assert any(t == "full" for t in tiers.values())
    assert 40 in steps and 30 in steps


def test_backpressure_blocks_then_clears_after_publish(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    mgr = PushManager(backend, backpressure_minutes=0.001)
    mgr.mark_started()
    time.sleep(0.08)
    with pytest.raises(BackpressureError):
        mgr.check_backpressure()
    ckpt = make_fake_checkpoint(tmp_path / "local", step=3)
    mgr.publish(ckpt, step=3)
    mgr.check_backpressure()  # fresh success resets the age


def test_transient_upload_failures_are_retried(tmp_path):
    class FlakyLocal(LocalBackend):
        def __init__(self, root, folder_failures=3, bytes_failures=2):
            super().__init__(root)
            self.folder_failures = folder_failures
            self.bytes_failures = bytes_failures

        @retry(**hub._RETRY)
        def upload_folder(self, local_dir, dest_prefix, message):
            if self.folder_failures:
                self.folder_failures -= 1
                raise ConnectionError("simulated 50% network failure")
            return super().upload_folder(local_dir, dest_prefix, message)

        @retry(**hub._RETRY)
        def upload_bytes(self, data, dest_path, message):
            if self.bytes_failures:
                self.bytes_failures -= 1
                raise ConnectionError("simulated 50% network failure")
            return super().upload_bytes(data, dest_path, message)

    backend = FlakyLocal(tmp_path / "hub")
    ckpt = make_fake_checkpoint(tmp_path / "local", step=9)
    mgr = PushManager(backend)
    mgr.publish(ckpt, step=9)
    assert mgr.resolve_latest() == 9
    assert backend.folder_failures == 0 and backend.bytes_failures == 0


def test_heartbeat_is_bounded(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    mgr = PushManager(backend)
    mgr.write_heartbeat({"step": 1, "run_id": "r", "loss_ema": 1.0})
    assert '"step": 1' in backend.read_bytes("heartbeat.json").decode()
    with pytest.raises(ValueError, match="1 KB"):
        mgr.write_heartbeat({"blob": "x" * 2048})


def test_upload_probe_cleans_up(tmp_path):
    backend = LocalBackend(tmp_path / "hub")
    seconds = probe_upload(backend, size_bytes=4096)
    assert seconds >= 0
    assert backend.list_files("_probe/") == []


def test_hf_list_files_ignores_folder_nodes(monkeypatch):
    """`list_repo_tree` yields directories too; they are not remote files."""
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    from t4llm.hub import HFBackend

    class FakeApi:
        def list_repo_tree(self, *args, **kwargs):
            return [
                RepoFolder(path="ckpt-00000010", oid="tree-root"),
                RepoFolder(path="ckpt-00000010/tokenizer", oid="tree-tokenizer"),
                RepoFile(
                    path="ckpt-00000010/adapter_model.safetensors",
                    size=4096,
                    oid="blob-adapter",
                    lfs={"size": 4096, "oid": "a" * 64, "pointerSize": 134},
                ),
                RepoFile(
                    path="ckpt-00000010/manifest.json",
                    size=123,
                    oid="blob-manifest",
                ),
            ]

    # Bypass __init__ so the test never needs a real token or network.
    backend = object.__new__(HFBackend)
    backend.repo_id = "unit/test"
    backend.repo_type = "model"
    backend._token = "not-a-real-token"
    backend.api = FakeApi()

    files = backend.list_files("ckpt-00000010/")
    assert [f.path for f in files] == [
        "ckpt-00000010/adapter_model.safetensors",
        "ckpt-00000010/manifest.json",
    ]
    assert files[0].sha256 == "a" * 64
    assert files[1].size == 123
