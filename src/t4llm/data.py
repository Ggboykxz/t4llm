"""Immutable packed data shards, deterministic global order and O(1) cursor (§6).

Build happens offline, once.  Training never tokenizes and never shuffles at
runtime.  The global permutation is materialised into the shard contents; the
cursor is therefore a contiguous (shard, sample) position.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .state import (
    Cursor,
    StateError,
    atomic_write_text,
    canonical_json,
    sha256_bytes,
    sha256_file,
    utcnow,
)

INDEX_FORMAT = "t4llm-packed-v1"
MASK64 = (1 << 64) - 1
MINHASH_HASHES = 64
MINHASH_BANDS = 16
MINHASH_ROWS = 4
MINHASH_THRESHOLD = 0.70
SHINGLE = 5


class DataError(StateError):
    pass


class DataExhausted(DataError):
    pass


def _require_license(license_name: str) -> str:
    lic = (license_name or "").strip()
    if not lic or lic.lower() in {"unknown", "none", "n/a", "unlicensed"}:
        raise DataError(
            "every document/source must declare a licence; refusing to build data "
            "whose licence cannot be stated"
        )
    return lic


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _stable_u64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


class MinHashDeduper:
    """Deterministic MinHash + banded LSH.  No Python hash randomisation."""

    def __init__(self, seed: int = 3407):
        rng = np.random.default_rng(seed)
        self._a = (rng.integers(1, 1 << 63, size=MINHASH_HASHES, dtype=np.uint64) * 2 + 1)
        self._b = rng.integers(0, 1 << 63, size=MINHASH_HASHES, dtype=np.uint64)
        self._buckets: list[dict[bytes, list[int]]] = [
            {} for _ in range(MINHASH_BANDS)
        ]
        self._signatures: list[np.ndarray] = []

    @staticmethod
    def _shingles(tokens: list[int]) -> np.ndarray:
        if not tokens:
            return np.array([_stable_u64(b"<empty>")], dtype=np.uint64)
        if len(tokens) < SHINGLE:
            return np.array([_stable_u64(",".join(map(str, tokens)).encode())], dtype=np.uint64)
        t = np.asarray(tokens, dtype=np.uint64)
        width = len(t) - SHINGLE + 1
        h = np.zeros(width, dtype=np.uint64)
        for k in range(SHINGLE):
            h = (h * np.uint64(1000003) + t[k : k + width]) & np.uint64(MASK64)
        return h

    def signature(self, tokens: list[int]) -> np.ndarray:
        shingles = self._shingles(tokens)
        sig = np.full(MINHASH_HASHES, MASK64, dtype=np.uint64)
        for i in range(MINHASH_HASHES):
            vals = (shingles * self._a[i] + self._b[i]) & np.uint64(MASK64)
            sig[i] = vals.min()
        return sig

    def is_near_duplicate(self, sig: np.ndarray) -> bool:
        candidates: set[int] = set()
        for band in range(MINHASH_BANDS):
            key = sig[band * MINHASH_ROWS : (band + 1) * MINHASH_ROWS].tobytes()
            candidates.update(self._buckets[band].get(key, ()))
        for idx in candidates:
            if float(np.mean(self._signatures[idx] == sig)) >= MINHASH_THRESHOLD:
                return True
        for band in range(MINHASH_BANDS):
            key = sig[band * MINHASH_ROWS : (band + 1) * MINHASH_ROWS].tobytes()
            self._buckets[band].setdefault(key, []).append(len(self._signatures))
        self._signatures.append(sig)
        return False


def _tokenize_batch(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    enc = tokenizer(texts, add_special_tokens=False)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if ids and isinstance(ids[0], int):
        ids = [ids]
    return [list(map(int, row)) for row in ids]


def _eos_id(tokenizer: Any) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise DataError("tokenizer has no eos_token_id; refusing to pack without EOS")
    return int(eos)


def iter_jsonl_docs(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            for key in ("text", "source", "license"):
                if key not in doc:
                    raise DataError(f"{p}:{lineno}: missing {key!r}")
            yield doc


def _shard_hash(shard_dir: Path) -> str:
    """Content hash of Arrow payload only; metadata files are not content."""
    h = hashlib.sha256()
    files = sorted(shard_dir.glob("*.arrow"))
    if not files:
        raise DataError(f"shard has no Arrow file: {shard_dir}")
    for p in files:
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def _build_index(
    root: Path,
    *,
    seq_len: int,
    seed: int,
    tokenizer_name: str,
    total_samples: int,
    total_tokens: int,
    stream_sha256: str,
    shard_specs: list[dict[str, Any]],
    stats: dict[str, Any],
    sources: dict[str, Any],
    synthetic: bool,
) -> dict[str, Any]:
    index: dict[str, Any] = {
        "format": INDEX_FORMAT,
        "seq_len": seq_len,
        "seed": seed,
        "tokenizer": tokenizer_name,
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "stream_sha256": stream_sha256,
        "shards": shard_specs,
        "stats": stats,
        "sources": sources,
        "synthetic": synthetic,
        "created_utc": utcnow(),
    }
    # created_utc is informational and must not make an identical rebuild a new
    # revision; the revision is content-addressed.
    body = {k: v for k, v in index.items() if k != "created_utc"}
    index["data_revision"] = sha256_bytes(canonical_json(body).encode("utf-8"))
    atomic_write_text(root / "index.json", json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index


def load_index(path_or_dir: str | Path) -> dict[str, Any]:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "index.json"
    try:
        index = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read data index {p}: {exc}") from exc
    if index.get("format") != INDEX_FORMAT:
        raise DataError(f"unsupported data format: {index.get('format')!r}")
    claimed = index.get("data_revision")
    body = dict(index)
    body.pop("data_revision", None)
    body.pop("created_utc", None)
    actual = sha256_bytes(canonical_json(body).encode("utf-8"))
    if claimed != actual:
        raise DataError(
            f"data revision mismatch: index claims {claimed!r}, content hashes to {actual!r}"
        )
    total = 0
    expected_start = 0
    for i, spec in enumerate(index["shards"]):
        if spec["index"] != i:
            raise DataError(f"shard index {spec['index']} != position {i}")
        if spec["sample_start"] != expected_start:
            raise DataError(f"shard {i} sample_start is not contiguous")
        if not isinstance(spec.get("files"), dict) or not spec["files"]:
            raise DataError(f"shard {i} has no file inventory")
        expected_start += int(spec["samples"])
        total += int(spec["samples"])
    if total != int(index["total_samples"]):
        raise DataError("shard sample counts do not sum to total_samples")
    if total * int(index["seq_len"]) != int(index["total_tokens"]):
        raise DataError("total_tokens != total_samples * seq_len")
    return index


def publish_data(out_dir: str | Path, backend: Any) -> str:
    """Upload shards first, index last.  Returns the immutable data revision."""
    root = Path(out_dir)
    index = load_index(root)
    rev = index["data_revision"]
    prefix = f"rev-{rev}"
    for spec in index["shards"]:
        name = spec["file"]
        remote = f"{prefix}/{name}/"
        existing = backend.list_files(remote)
        if existing:
            continue  # content-addressed path already complete; never rewrite
        backend.upload_folder(root / name, f"{prefix}/{name}", f"data {rev} {name}")
    backend.upload_bytes(
        (json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        f"{prefix}/index.json",
        f"data index {rev}",
    )
    # Also refresh a small pointer for humans; training pins rev-<hash> directly.
    backend.upload_bytes((rev + "\n").encode("utf-8"), "LATEST_DATA", f"data {rev}")
    return rev


def fetch_data(
    backend: Any,
    revision: str,
    cache_dir: str | Path,
    *,
    expected_seq_len: int | None = None,
) -> tuple[dict[str, Any], Path]:
    """Download and verify an immutable revision.  Cached shards are rechecked."""
    if not revision or not re.fullmatch(r"[0-9a-f]{64}", revision):
        raise DataError(f"data revision must be a 64-hex content hash, got {revision!r}")
    prefix = f"rev-{revision}"
    dest = Path(cache_dir) / prefix
    dest.mkdir(parents=True, exist_ok=True)
    raw = backend.read_bytes(f"{prefix}/index.json")
    index_path = dest / "index.json"
    atomic_write_text(index_path, raw.decode("utf-8"))
    index = load_index(index_path)
    if index["data_revision"] != revision:
        raise DataError("downloaded index revision does not match requested revision")
    if expected_seq_len is not None and int(index["seq_len"]) != int(expected_seq_len):
        raise DataError(
            f"data seq_len {index['seq_len']} != training seq_len {expected_seq_len}; "
            "rebuild the data rather than silently changing the effective batch"
        )
    for spec in index["shards"]:
        shard_dir = dest / spec["file"]
        if not shard_dir.exists() or _shard_hash(shard_dir) != spec["sha256"]:
            if shard_dir.exists():
                shutil.rmtree(shard_dir)
            shard_dir.mkdir(parents=True, exist_ok=True)
            for rel, entry in spec["files"].items():
                backend.download_to(f"{prefix}/{spec['file']}/{rel}", shard_dir / rel)
                p = shard_dir / rel
                if p.stat().st_size != int(entry["bytes"]) or sha256_file(p) != entry["sha256"]:
                    raise DataError(f"corrupt downloaded shard file: {spec['file']}/{rel}")
            actual = _shard_hash(shard_dir)
            if actual != spec["sha256"]:
                raise DataError(f"downloaded shard hash mismatch: {spec['file']}")
    return index, dest


# --------------------------------------------------------------------------- #
# Offline builder
# --------------------------------------------------------------------------- #
def build_packed(
    docs: Iterable[dict[str, Any]],
    out_dir: str | Path,
    tokenizer: Any,
    *,
    seq_len: int,
    seed: int = 3407,
    max_tokens_per_shard: int = 50_000_000,
    near_dedup: bool = True,
    tokenizer_name: str | None = None,
    synthetic: bool = False,
    progress_every: int = 1000,
) -> dict[str, Any]:
    if seq_len <= 0:
        raise DataError("seq_len must be positive")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for p in root.glob("shard-*"):
        shutil.rmtree(p)
    eos = _eos_id(tokenizer)
    deduper = MinHashDeduper(seed)
    exact: set[str] = set()
    tmp_tokens = root / ".build-tokens.u32"
    stats = {
        "raw_docs": 0,
        "exact_dups": 0,
        "near_dups": 0,
        "empty_docs": 0,
        "kept_docs": 0,
        "dropped_tail_tokens": 0,
        "raw_tokens": 0,
    }
    sources: dict[str, dict[str, Any]] = {}
    total_tokens = 0

    with open(tmp_tokens, "wb") as fh:
        batch: list[dict[str, Any]] = []
        batch_size = 128

        def flush() -> None:
            nonlocal total_tokens
            if not batch:
                return
            texts = [d["text"] for d in batch]
            token_rows = _tokenize_batch(tokenizer, texts)
            for doc, ids in zip(batch, token_rows):
                tokenised = list(ids) + [eos]
                stats["raw_tokens"] += len(tokenised)
                if not ids:
                    stats["empty_docs"] += 1
                    continue
                if near_dedup and deduper.is_near_duplicate(deduper.signature(ids)):
                    stats["near_dups"] += 1
                    continue
                arr = np.asarray(tokenised, dtype=np.uint32)
                fh.write(arr.tobytes(order="C"))
                total_tokens += len(tokenised)
                stats["kept_docs"] += 1
                src = str(doc["source"])
                lic = _require_license(str(doc["license"]))
                rec = sources.setdefault(src, {"license": lic, "docs": 0, "tokens": 0})
                if rec["license"] != lic:
                    raise DataError(f"source {src!r} has conflicting licences")
                rec["docs"] += 1
                rec["tokens"] += len(tokenised)
            batch.clear()

        for doc in docs:
            stats["raw_docs"] += 1
            text = _normalise(str(doc.get("text", "")))
            _require_license(str(doc.get("license", "")))
            if not text:
                stats["empty_docs"] += 1
                continue
            h = sha256_bytes(text.encode("utf-8"))
            if h in exact:
                stats["exact_dups"] += 1
                continue
            exact.add(h)
            batch.append(doc)
            if len(batch) >= batch_size:
                flush()
                if progress_every and stats["raw_docs"] % progress_every == 0:
                    print(
                        f"data: raw={stats['raw_docs']} kept={stats['kept_docs']} "
                        f"tokens={total_tokens/1e6:.1f}M"
                    )
        flush()
        fh.flush()
        os.fsync(fh.fileno())

    stats["dropped_tail_tokens"] = total_tokens % seq_len
    num_samples = total_tokens // seq_len
    if num_samples == 0:
        raise DataError("no complete packed samples; add data or reduce seq_len")
    stream_sha = sha256_file(tmp_tokens)
    # Permutation depends on seed AND the immutable token stream, not wall time.
    perm_seed = seed ^ int.from_bytes(bytes.fromhex(stream_sha)[:8], "little")
    perm = np.random.default_rng(perm_seed).permutation(num_samples)
    mm = np.memmap(tmp_tokens, dtype=np.uint32, mode="r", shape=(num_samples, seq_len))

    from datasets import Dataset, Features, Sequence, Value

    features = Features({"input_ids": Sequence(Value("uint32"))})
    shard_samples = max(1, int(max_tokens_per_shard) // seq_len)
    shard_specs: list[dict[str, Any]] = []
    sample_cursor = 0
    shard_idx = 0
    while sample_cursor < num_samples:
        n = min(shard_samples, num_samples - sample_cursor)
        indices = perm[sample_cursor : sample_cursor + n]
        name = f"shard-{shard_idx:05d}"
        shard_dir = root / name
        shard_dir.mkdir(parents=True, exist_ok=True)

        def gen() -> Iterator[dict[str, Any]]:
            for idx in indices:
                yield {"input_ids": [int(x) for x in mm[int(idx)]]}

        ds = Dataset.from_generator(
            gen,
            features=features,
            cache_dir=str(root / ".build-cache"),
        )
        ds.save_to_disk(str(shard_dir))
        ds.cleanup_cache_files()
        files: dict[str, dict[str, Any]] = {}
        for p in sorted(shard_dir.iterdir()):
            if p.is_file():
                files[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        shard_specs.append(
            {
                "index": shard_idx,
                "file": name,
                "sample_start": sample_cursor,
                "samples": n,
                "tokens": n * seq_len,
                "sha256": _shard_hash(shard_dir),
                "files": files,
            }
        )
        sample_cursor += n
        shard_idx += 1
        print(f"data: wrote {name} samples={n} tokens={n*seq_len/1e6:.1f}M")

    stats["raw_tokens"] = int(stats["raw_tokens"])
    stats["tokenizer_eos_id"] = eos
    index = _build_index(
        root,
        seq_len=seq_len,
        seed=seed,
        tokenizer_name=tokenizer_name or getattr(tokenizer, "name_or_path", "unknown"),
        total_samples=num_samples,
        total_tokens=num_samples * seq_len,
        stream_sha256=stream_sha,
        shard_specs=shard_specs,
        stats=stats,
        sources=sources,
        synthetic=synthetic,
    )
    del mm
    tmp_tokens.unlink(missing_ok=True)
    shutil.rmtree(root / ".build-cache", ignore_errors=True)
    return index


def build_synthetic_shards(
    out_dir: str | Path,
    *,
    vocab_size: int = 128,
    seq_len: int = 64,
    num_samples: int = 2000,
    seed: int = 3407,
    max_samples_per_shard: int = 1000,
) -> dict[str, Any]:
    """Test/chaos fixture only.  The trainer refuses synthetic data without a flag."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for p in root.glob("shard-*"):
        shutil.rmtree(p)
    from datasets import Dataset, Features, Sequence, Value

    rng = np.random.default_rng(seed)
    features = Features({"input_ids": Sequence(Value("uint32"))})
    shard_specs: list[dict[str, Any]] = []
    for start in range(0, num_samples, max_samples_per_shard):
        n = min(max_samples_per_shard, num_samples - start)
        samples = rng.integers(0, vocab_size, size=(n, seq_len), dtype=np.uint32)
        name = f"shard-{len(shard_specs):05d}"
        shard_dir = root / name
        shard_dir.mkdir(parents=True, exist_ok=True)
        ds = Dataset.from_dict({"input_ids": samples.tolist()}, features=features)
        ds.save_to_disk(str(shard_dir))
        files = {
            p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(shard_dir.iterdir())
            if p.is_file()
        }
        shard_specs.append(
            {
                "index": len(shard_specs),
                "file": name,
                "sample_start": start,
                "samples": n,
                "tokens": n * seq_len,
                "sha256": _shard_hash(shard_dir),
                "files": files,
            }
        )
    index = _build_index(
        root,
        seq_len=seq_len,
        seed=seed,
        tokenizer_name=f"synthetic-vocab-{vocab_size}",
        total_samples=num_samples,
        total_tokens=num_samples * seq_len,
        stream_sha256="synthetic",
        shard_specs=shard_specs,
        stats={
            "raw_docs": num_samples,
            "exact_dups": 0,
            "near_dups": 0,
            "kept_docs": num_samples,
            "synthetic": True,
        },
        sources={"synthetic": {"license": "test-only", "docs": num_samples}},
        synthetic=True,
    )
    return index


def write_data_report(index: dict[str, Any], docs_dir: str | Path = "docs") -> Path:
    p = Path(docs_dir) / "DATA.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    s = index["stats"]
    raw = max(1, int(s.get("raw_docs", 0)))
    dup = int(s.get("exact_dups", 0)) + int(s.get("near_dups", 0))
    lines = [
        "# Data report",
        "",
        f"Generated: {index.get('created_utc')}",
        f"Data revision: `{index['data_revision']}`",
        f"Tokenizer: `{index['tokenizer']}`",
        f"Sequence length: {index['seq_len']}",
        f"Packed samples: {index['total_samples']:,}",
        f"Training tokens: {index['total_tokens']:,}",
        "",
        "## Corpus funnel",
        "",
        "| Stage | Count |",
        "|---|---:|",
        f"| Raw documents | {raw:,} |",
        f"| Exact duplicates removed | {int(s.get('exact_dups', 0)):,} |",
        f"| Near duplicates removed (MinHash) | {int(s.get('near_dups', 0)):,} |",
        f"| Empty documents | {int(s.get('empty_docs', 0)):,} |",
        f"| Documents kept | {int(s.get('kept_docs', 0)):,} |",
        f"| Duplicate rate | {100.0*dup/raw:.2f}% |",
        f"| Dropped tail tokens | {int(s.get('dropped_tail_tokens', 0)):,} |",
        "",
        "## Sources and licences",
        "",
        "| Source | Licence | Docs | Tokens |",
        "|---|---|---:|---:|",
    ]
    for name, rec in sorted(index.get("sources", {}).items()):
        lines.append(
            f"| {name} | {rec.get('license', 'MISSING')} | "
            f"{int(rec.get('docs', 0)):,} | {int(rec.get('tokens', 0)):,} |"
        )
    lines += [
        "",
        "## Packing note",
        "",
        "EOS is inserted between documents.  Standard causal attention is used across",
        "packed document boundaries, so cross-document attention contamination is",
        "explicitly accepted for CPT (ADR-0003).  SFT/DPO do not pack.",
        "",
    ]
    atomic_write_text(p, "\n".join(lines))
    return p


# --------------------------------------------------------------------------- #
# Cursor
# --------------------------------------------------------------------------- #
class PackedCursor:
    """O(1) seek over immutable packed shards.  One shard resident at a time."""

    def __init__(
        self,
        index: dict[str, Any],
        root: str | Path,
        *,
        shard_idx: int = 0,
        sample_idx: int = 0,
        consumed_tokens: int = 0,
        verify_hashes: bool = False,
    ):
        self.index = index
        self.root = Path(root)
        self.seq_len = int(index["seq_len"])
        self._shards = index["shards"]
        self._cache_idx: int | None = None
        self._cache = None
        if verify_hashes:
            for spec in self._shards:
                if _shard_hash(self.root / spec["file"]) != spec["sha256"]:
                    raise DataError(f"shard hash mismatch: {spec['file']}")
        self.shard_idx = 0
        self.sample_idx = 0
        self.consumed_tokens = 0
        self.seek(shard_idx, sample_idx, consumed_tokens)

    @classmethod
    def from_index(
        cls, index_path: str | Path, *, verify_hashes: bool = False
    ) -> "PackedCursor":
        p = Path(index_path)
        root = p.parent if p.is_file() else p
        index = load_index(p)
        return cls(index, root, verify_hashes=verify_hashes)

    def _shard(self, idx: int):
        if idx >= len(self._shards):
            raise DataExhausted("no more shards")
        if self._cache_idx != idx:
            from datasets import load_from_disk

            self._cache = load_from_disk(str(self.root / self._shards[idx]["file"]))
            self._cache_idx = idx
        return self._cache

    def _sample_start(self, shard_idx: int) -> int:
        return int(self._shards[shard_idx]["sample_start"])

    def seek(self, shard_idx: int, sample_idx: int, consumed_tokens: int) -> None:
        if not (0 <= shard_idx < len(self._shards)):
            if shard_idx == len(self._shards) and sample_idx == 0:
                if consumed_tokens != int(self.index["total_tokens"]):
                    raise DataError(
                        "end-of-data cursor must have consumed exactly total_tokens"
                    )
                self.shard_idx, self.sample_idx = shard_idx, 0
                self.consumed_tokens = int(consumed_tokens)
                return
            raise DataError(f"shard_idx {shard_idx} out of range")
        spec = self._shards[shard_idx]
        if not (0 <= sample_idx <= int(spec["samples"])):
            raise DataError(f"sample_idx {sample_idx} out of range for shard {shard_idx}")
        expected = (self._sample_start(shard_idx) + sample_idx) * self.seq_len
        if consumed_tokens != expected:
            raise DataError(
                f"cursor consumed_tokens {consumed_tokens} disagrees with position "
                f"(expected {expected}); refusing to guess"
            )
        self.shard_idx = shard_idx
        self.sample_idx = sample_idx
        self.consumed_tokens = consumed_tokens

    def cursor(self) -> Cursor:
        return Cursor(
            data_revision=self.index["data_revision"],
            shard_idx=self.shard_idx,
            sample_idx=self.sample_idx,
            consumed_tokens=self.consumed_tokens,
        )

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, Cursor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rows: list[np.ndarray] = []
        collected = 0
        while collected < batch_size:
            if self.shard_idx >= len(self._shards):
                raise DataExhausted("data exhausted; stop this run rather than repeat silently")
            spec = self._shards[self.shard_idx]
            n = int(spec["samples"])
            if self.sample_idx >= n:
                self.shard_idx += 1
                self.sample_idx = 0
                continue
            take = min(batch_size - collected, n - self.sample_idx)
            ds = self._shard(self.shard_idx)
            chunk = ds.select(range(self.sample_idx, self.sample_idx + take))
            arr = np.asarray(chunk["input_ids"], dtype=np.uint32)
            if arr.ndim != 2 or arr.shape[1] != self.seq_len:
                raise DataError("shard returned a malformed sequence")
            rows.append(arr)
            collected += take
            self.sample_idx += take
            self.consumed_tokens += take * self.seq_len
        return np.concatenate(rows, axis=0), self.cursor()

    @property
    def remaining_samples(self) -> int:
        total = int(self.index["total_samples"])
        pos = self._sample_start(self.shard_idx) + self.sample_idx if self.shard_idx < len(self._shards) else total
        return max(0, total - pos)
