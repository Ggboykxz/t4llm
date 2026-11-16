"""Data contract tests (§6): licence gate, dedup, deterministic packing, cursor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from t4llm.data import (
    DataError,
    DataExhausted,
    PackedCursor,
    build_packed,
    build_synthetic_shards,
    fetch_data,
    iter_jsonl_docs,
    load_index,
    publish_data,
    write_data_report,
)
from t4llm.hub import LocalBackend, FatalHubError

from helpers import make_fake_checkpoint  # noqa: F401  (keeps helpers importable)


def docs(n: int = 5):
    # Tokens are drawn from the tiny fixture vocabulary so the WordLevel
    # tokenizer does not collapse everything to <unk>.
    out = []
    for i in range(n):
        words = " ".join(f"tok{5 + i * 20 + j}" for j in range(20))
        out.append({"text": words, "source": "unit-test", "license": "CC0-1.0"})
    return out


def test_synthetic_is_deterministic(tmp_path):
    a = build_synthetic_shards(
        tmp_path / "a", seq_len=16, num_samples=120, seed=7, max_samples_per_shard=50
    )
    b = build_synthetic_shards(
        tmp_path / "b", seq_len=16, num_samples=120, seed=7, max_samples_per_shard=50
    )
    assert a["data_revision"] == b["data_revision"]
    assert a["shards"][0]["sha256"] == b["shards"][0]["sha256"]
    assert a["total_tokens"] == a["total_samples"] * 16
    assert sum(s["samples"] for s in a["shards"]) == 120
    assert [s["sample_start"] for s in a["shards"]] == [0, 50, 100]


def test_licence_is_mandatory(tmp_path):
    bad = [{"text": "hello world", "source": "s", "license": "unknown"}]
    tokenizer = _tiny_tokenizer(tmp_path)
    with pytest.raises(DataError, match="licence"):
        build_packed(bad, tmp_path / "packed", tokenizer, seq_len=4)


def test_jsonl_reader_rejects_missing_fields(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text('{"text": "hello", "source": "s"}\n')
    with pytest.raises(DataError, match="missing 'license'"):
        list(iter_jsonl_docs(p))


def test_packer_dedups_and_records_sources(tmp_path):
    tokenizer = _tiny_tokenizer(tmp_path)
    corpus = docs(5)
    corpus.append(dict(corpus[0]))  # exact duplicate
    corpus.append(
        {
            "text": corpus[1]["text"] + " tok100 tok101",
            "source": "unit-test",
            "license": "CC0-1.0",
        }
    )  # near duplicate
    index = build_packed(
        corpus,
        tmp_path / "packed",
        tokenizer,
        seq_len=8,
        near_dedup=True,
        max_tokens_per_shard=64,
    )
    assert index["stats"]["exact_dups"] == 1
    assert index["stats"]["near_dups"] == 1
    assert index["sources"]["unit-test"]["license"] == "CC0-1.0"
    assert index["total_tokens"] % 8 == 0
    assert len(index["shards"]) >= 2
    report = write_data_report(index, tmp_path / "docs")
    assert "CC0-1.0" in report.read_text()


def test_packed_cursor_order_matches_shards(tmp_path):
    index = build_synthetic_shards(
        tmp_path / "data", seq_len=32, num_samples=250, seed=11, max_samples_per_shard=100
    )
    cursor = PackedCursor(index, tmp_path / "data")
    seen = []
    while cursor.remaining_samples:
        batch, cur = cursor.next_batch(min(7, cursor.remaining_samples))
        seen.append(np.asarray(batch))
        assert cur.consumed_tokens == cursor.consumed_tokens
        assert cur.sample_idx == cursor.sample_idx
    all_rows = np.concatenate(seen, axis=0)
    assert all_rows.shape == (250, 32)
    assert cursor.consumed_tokens == index["total_tokens"]

    from datasets import load_from_disk

    expected = np.concatenate(
        [
            np.asarray(load_from_disk(str(tmp_path / "data" / s["file"]))["input_ids"])
            for s in index["shards"]
        ],
        axis=0,
    )
    assert np.array_equal(all_rows, expected)


def test_cursor_seek_is_exact_and_o1(tmp_path):
    index = build_synthetic_shards(
        tmp_path / "data", seq_len=16, num_samples=120, seed=3, max_samples_per_shard=50
    )
    cursor = PackedCursor(index, tmp_path / "data")
    cursor.seek(1, 10, (50 + 10) * 16)
    batch, cur = cursor.next_batch(5)
    assert cur.shard_idx == 1 and cur.sample_idx == 15
    assert cur.consumed_tokens == 65 * 16
    assert np.asarray(batch).shape == (5, 16)
    with pytest.raises(DataError, match="disagrees"):
        cursor.seek(1, 10, 12345)
    cursor.seek(len(index["shards"]), 0, index["total_tokens"])
    with pytest.raises(DataExhausted):
        cursor.next_batch(1)


def test_index_tamper_is_detected(tmp_path):
    build_synthetic_shards(tmp_path / "data", seq_len=8, num_samples=20)
    p = tmp_path / "data" / "index.json"
    text = p.read_text().replace('"total_tokens": 160', '"total_tokens": 161')
    p.write_text(text)
    with pytest.raises(DataError, match="revision mismatch"):
        load_index(p)


def test_publish_then_fetch_roundtrip(tmp_path):
    build_synthetic_shards(
        tmp_path / "data", seq_len=16, num_samples=90, seed=5, max_samples_per_shard=40
    )
    backend = LocalBackend(tmp_path / "hub")
    rev = publish_data(tmp_path / "data", backend)
    assert backend.read_bytes("LATEST_DATA").decode().strip() == rev
    assert backend.list_files(f"rev-{rev}/index.json")
    index, root = fetch_data(backend, rev, tmp_path / "cache", expected_seq_len=16)
    assert index["data_revision"] == rev
    cursor = PackedCursor(index, root)
    batch, _ = cursor.next_batch(3)
    assert np.asarray(batch).shape == (3, 16)


def test_publish_index_last_on_failure(tmp_path):
    build_synthetic_shards(tmp_path / "data", seq_len=8, num_samples=30)

    class NoIndex(LocalBackend):
        def upload_bytes(self, data, dest_path, message):
            if dest_path.endswith("index.json") or dest_path == "LATEST_DATA":
                raise FatalHubError("simulated index commit failure")
            return super().upload_bytes(data, dest_path, message)

    backend = NoIndex(tmp_path / "hub")
    with pytest.raises(FatalHubError):
        publish_data(tmp_path / "data", backend)
    assert backend.list_files("rev-") and not backend.list_files("LATEST_DATA")
    # A reader can never see a revision pointer without its index.
    assert not any(f.path.endswith("index.json") for f in backend.list_files("rev-"))


def test_fetch_rejects_wrong_seq_len_and_bad_revision(tmp_path):
    build_synthetic_shards(tmp_path / "data", seq_len=8, num_samples=20)
    backend = LocalBackend(tmp_path / "hub")
    rev = publish_data(tmp_path / "data", backend)
    with pytest.raises(DataError, match="seq_len"):
        fetch_data(backend, rev, tmp_path / "cache", expected_seq_len=16)
    with pytest.raises(DataError, match="64-hex"):
        fetch_data(backend, "../../etc", tmp_path / "cache")


def test_fetch_repairs_corrupt_local_cache(tmp_path):
    build_synthetic_shards(tmp_path / "data", seq_len=8, num_samples=20)
    backend = LocalBackend(tmp_path / "hub")
    rev = publish_data(tmp_path / "data", backend)
    index, root = fetch_data(backend, rev, tmp_path / "cache")
    shard = root / index["shards"][0]["file"]
    arrow = next(shard.glob("*.arrow"))
    arrow.write_bytes(b"corrupted")
    index2, root2 = fetch_data(backend, rev, tmp_path / "cache")
    assert root2 == root
    repaired = root2 / index2["shards"][0]["file"] / arrow.name
    assert repaired.is_file()
    assert repaired.read_bytes() != b"corrupted"


def test_fetch_detects_corrupt_remote_shard(tmp_path):
    build_synthetic_shards(tmp_path / "data", seq_len=8, num_samples=20)
    backend = LocalBackend(tmp_path / "hub")
    rev = publish_data(tmp_path / "data", backend)
    remote = next(p for p in (tmp_path / "hub" / f"rev-{rev}").rglob("*.arrow"))
    remote.write_bytes(b"remote bitrot")
    with pytest.raises(DataError, match="corrupt downloaded shard|hash mismatch"):
        fetch_data(backend, rev, tmp_path / "cache2")


def _tiny_tokenizer(tmp_path: Path):
    from transformers import AutoTokenizer

    from t4llm.fixtures import make_tiny_model

    return AutoTokenizer.from_pretrained(str(make_tiny_model(tmp_path / "tiny")))
