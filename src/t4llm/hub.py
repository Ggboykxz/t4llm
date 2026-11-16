"""Hub protocol: atomic publish, verification, retention, retries (§8).

Production transport is the Hugging Face HTTP API.  A local-directory backend
implements the same commit/verify semantics for CPU tests and the chaos test;
it is a test transport, never the durable store for a real run.

The token is read from the process environment and passed per call.  It is
never logged, never written to a file and never stored by huggingface_hub.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .state import (
    MANIFEST_NAME,
    SUMS_NAME,
    CheckpointManifest,
    ManifestError,
    atomic_write_bytes,
    canonical_json,
    read_manifest,
    sha256_bytes,
    sha256_file,
    utcnow,
    validate_checkpoint_dir,
)

CKPT_PREFIX = "ckpt-"
LATEST = "LATEST"
HEARTBEAT = "heartbeat.json"
METRICS = "metrics.jsonl"


class HubError(RuntimeError):
    pass


class FatalHubError(HubError):
    """401/403/404/schema conflict/immutability violation.  Do not retry."""


class RetryableHubError(HubError):
    pass


class CheckpointExistsError(FatalHubError):
    pass


class CorruptCheckpointError(HubError):
    pass


class BackpressureError(HubError):
    pass


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None = None


class Backend(Protocol):
    name: str

    def list_files(self, prefix: str = "") -> list[RemoteFile]: ...
    def read_bytes(self, path: str) -> bytes: ...
    def download_to(self, path: str, dest: str | Path) -> None: ...
    def upload_folder(self, local_dir: str | Path, dest_prefix: str, message: str) -> None: ...
    def upload_bytes(self, data: bytes, dest_path: str, message: str) -> None: ...
    def delete_paths(self, paths: Iterable[str], message: str) -> None: ...


# --------------------------------------------------------------------------- #
# Local backend (tests / chaos only)
# --------------------------------------------------------------------------- #
class LocalBackend:
    """Directory-backed store with atomic folder commits.

    A crash during upload leaves files under .staging, which list_files()
    ignores.  The destination either does not exist or is complete.  Published
    checkpoints are immutable.
    """

    name = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._staging = self.root / ".staging"
        self._staging.mkdir(exist_ok=True)

    def _visible(self) -> list[Path]:
        out = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root)
            if rel.parts and rel.parts[0] == ".staging":
                continue
            out.append(rel)
        return sorted(out, key=lambda x: x.as_posix())

    def list_files(self, prefix: str = "") -> list[RemoteFile]:
        files = []
        for rel in self._visible():
            path = rel.as_posix()
            if not path.startswith(prefix):
                continue
            p = self.root / rel
            files.append(RemoteFile(path, p.stat().st_size, sha256_file(p)))
        return files

    def read_bytes(self, path: str) -> bytes:
        p = self.root / path
        if not p.is_file():
            raise FatalHubError(f"remote file not found: {path}")
        return p.read_bytes()

    def download_to(self, path: str, dest: str | Path) -> None:
        src = self.root / path
        if not src.is_file():
            raise FatalHubError(f"remote file not found: {path}")
        d = Path(dest)
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(f".{d.name}.part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, d)

    def upload_folder(self, local_dir: str | Path, dest_prefix: str, message: str) -> None:
        src = Path(local_dir)
        target = self.root / dest_prefix
        if target.exists():
            raise CheckpointExistsError(f"checkpoint already published: {dest_prefix}")
        staging = self._staging / uuid.uuid4().hex
        try:
            shutil.copytree(src, staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            # A single rename makes the whole directory appear atomically.
            os.replace(staging, target)
        except FileExistsError as exc:
            raise CheckpointExistsError(f"checkpoint already published: {dest_prefix}") from exc
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def upload_bytes(self, data: bytes, dest_path: str, message: str) -> None:
        atomic_write_bytes(self.root / dest_path, data)

    def delete_paths(self, paths: Iterable[str], message: str) -> None:
        touched: set[Path] = set()
        for path in paths:
            p = self.root / path
            if p.is_file():
                p.unlink()
                touched.add(p.parent)
        for parent in sorted(touched, key=lambda p: len(p.parts), reverse=True):
            try:
                parent.rmdir()
            except OSError:
                pass

    def repo_size(self) -> int:
        return sum(p.stat().st_size for p in self._visible_files())

    def _visible_files(self):
        for rel in self._visible():
            yield self.root / rel


# --------------------------------------------------------------------------- #
# Hugging Face backend
# --------------------------------------------------------------------------- #
def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise FatalHubError(
            "HF_TOKEN is not set.  Add it as a Colab Secret; do not write it to disk."
        )
    return token


def _status_code(exc: BaseException) -> int | None:
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, FatalHubError):
        return False
    if isinstance(exc, RetryableHubError):
        return True
    code = _status_code(exc)
    if code is not None:
        if code in (401, 403, 404, 409):
            return False
        return code in (408, 425, 429) or code >= 500
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name or "temporary" in name:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def _classify(exc: BaseException) -> BaseException:
    if isinstance(exc, HubError):
        return exc
    code = _status_code(exc)
    if code in (401, 403):
        return FatalHubError(f"authentication/authorisation failed (HTTP {code}); do not retry")
    if code == 404:
        return FatalHubError("remote repo/path not found (HTTP 404); do not retry")
    if code == 409:
        return FatalHubError("remote schema/commit conflict (HTTP 409); do not retry")
    if code is not None and (code in (408, 425, 429) or code >= 500):
        return RetryableHubError(f"transient Hub error HTTP {code}")
    if _is_retryable(exc):
        return RetryableHubError(f"transient Hub error: {type(exc).__name__}")
    return FatalHubError(f"Hub error: {type(exc).__name__}: {str(exc)[:300]}")


_RETRY = dict(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(8),
    wait=wait_exponential_jitter(initial=1, max=120),
    reraise=True,
)


class HFBackend:
    name = "hf"

    def __init__(self, repo_id: str, *, repo_type: str = "model", token: str | None = None):
        if not repo_id or "/" not in repo_id:
            raise FatalHubError(f"invalid repo_id: {repo_id!r}")
        self.repo_id = repo_id
        self.repo_type = repo_type
        self._token = token or _hf_token()
        from huggingface_hub import HfApi

        self.api = HfApi(token=self._token)

    def __repr__(self) -> str:  # never include the token
        return f"HFBackend(repo_id={self.repo_id!r}, repo_type={self.repo_type!r})"

    def ensure_repo(self, private: bool = True) -> None:
        try:
            self.api.repo_info(self.repo_id, repo_type=self.repo_type)
        except Exception as exc:
            if _status_code(exc) != 404:
                raise _classify(exc) from exc
            try:
                self.api.create_repo(
                    self.repo_id,
                    repo_type=self.repo_type,
                    private=private,
                    exist_ok=True,
                    token=self._token,
                )
            except Exception as create_exc:
                raise _classify(create_exc) from create_exc

    @retry(**_RETRY)
    def list_files(self, prefix: str = "") -> list[RemoteFile]:
        try:
            # ``list_repo_tree(recursive=True)`` still yields folder nodes as
            # well as files.  A folder has no bytes; treating it as a file
            # would make remote verification reject a perfectly good
            # checkpoint as "unexpected file".  Filter by type and by the
            # absence of the RepoFile attributes.
            try:
                from huggingface_hub.hf_api import RepoFolder
            except Exception:  # pragma: no cover - defensive across hub versions
                class RepoFolder:  # type: ignore[no-redef]
                    pass

            out: list[RemoteFile] = []
            for item in self.api.list_repo_tree(
                self.repo_id,
                recursive=True,
                expand=True,
                repo_type=self.repo_type,
                token=self._token,
            ):
                if isinstance(item, RepoFolder):
                    continue
                path = getattr(item, "path", None)
                if path is None:
                    continue
                if not path.startswith(prefix):
                    continue
                size = int(getattr(item, "size", 0) or 0)
                lfs = getattr(item, "lfs", None)
                sha = None
                if isinstance(lfs, dict):
                    sha = lfs.get("oid") or lfs.get("sha256")
                elif lfs is not None:
                    sha = getattr(lfs, "oid", None) or getattr(lfs, "sha256", None)
                out.append(RemoteFile(path, size, sha))
            return sorted(out, key=lambda f: f.path)
        except Exception as exc:
            raise _classify(exc) from exc

    @retry(**_RETRY)
    def read_bytes(self, path: str) -> bytes:
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(
                self.repo_id,
                path,
                repo_type=self.repo_type,
                token=self._token,
                force_download=True,
            )
            return Path(local).read_bytes()
        except Exception as exc:
            raise _classify(exc) from exc

    @retry(**_RETRY)
    def download_to(self, path: str, dest: str | Path) -> None:
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(
                self.repo_id,
                path,
                repo_type=self.repo_type,
                token=self._token,
                force_download=True,
            )
            d = Path(dest)
            d.parent.mkdir(parents=True, exist_ok=True)
            tmp = d.with_name(f".{d.name}.part")
            shutil.copyfile(local, tmp)
            os.replace(tmp, d)
        except Exception as exc:
            raise _classify(exc) from exc

    @retry(**_RETRY)
    def upload_folder(self, local_dir: str | Path, dest_prefix: str, message: str) -> None:
        src = Path(local_dir)
        try:
            self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(src),
                path_in_repo=dest_prefix,
                repo_type=self.repo_type,
                token=self._token,
                commit_message=message,
                create_pr=False,
            )
        except Exception as exc:
            raise _classify(exc) from exc

    @retry(**_RETRY)
    def upload_bytes(self, data: bytes, dest_path: str, message: str) -> None:
        try:
            self.api.upload_file(
                path_or_fileobj=data,
                path_in_repo=dest_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self._token,
                commit_message=message,
            )
        except Exception as exc:
            raise _classify(exc) from exc

    @retry(**_RETRY)
    def delete_paths(self, paths: Iterable[str], message: str) -> None:
        from huggingface_hub import CommitOperationDelete

        ops = [CommitOperationDelete(path_in_repo=p) for p in sorted(set(paths))]
        if not ops:
            return
        try:
            self.api.create_commit(
                self.repo_id,
                operations=ops,
                commit_message=message,
                repo_type=self.repo_type,
                token=self._token,
                create_pr=False,
            )
        except Exception as exc:
            raise _classify(exc) from exc

    def repo_size(self) -> int:
        try:
            info = self.api.repo_info(
                self.repo_id, repo_type=self.repo_type, token=self._token, files_metadata=True
            )
            return sum(int(s.size or 0) for s in (info.siblings or []))
        except Exception as exc:
            raise _classify(exc) from exc


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
def make_backend(cfg: dict[str, Any], kind: str = "model") -> Backend:
    """kind='model' (checkpoints) or 'dataset' (immutable data shards)."""
    if kind not in {"model", "dataset"}:
        raise ValueError(kind)
    local_root = os.environ.get("T4LLM_HUB_ROOT")
    if kind == "dataset":
        local_root = os.environ.get("T4LLM_DATA_ROOT") or local_root
    if local_root:
        return LocalBackend(local_root)
    repo = cfg.get("run", {}).get("hub_repo" if kind == "model" else "data_repo")
    if not repo:
        raise FatalHubError(
            f"run.{'hub_repo' if kind == 'model' else 'data_repo'} is not configured; "
            "fill it in configs/base.yaml or set T4LLM_HUB_ROOT for a test transport"
        )
    return HFBackend(repo, repo_type="model" if kind == "model" else "dataset")


def checkpoint_prefix(step: int) -> str:
    return f"{CKPT_PREFIX}{step:08d}"


def parse_checkpoint_step(path: str) -> int | None:
    first = path.split("/", 1)[0]
    if not first.startswith(CKPT_PREFIX):
        return None
    try:
        return int(first[len(CKPT_PREFIX) :])
    except ValueError:
        return None


@dataclass
class Cadence:
    full_seconds: float = 360.0
    weights_only_seconds: float = 180.0
    upload_seconds: float | None = None

    @classmethod
    def from_upload(cls, upload_seconds: float | None) -> "Cadence":
        if upload_seconds is None:
            return cls()
        full = max(360.0, 3.0 * upload_seconds)
        weights = 180.0 if full > 600.0 else 0.0
        return cls(full_seconds=full, weights_only_seconds=weights, upload_seconds=upload_seconds)


class PushManager:
    """Single-flight publisher with verification, backpressure and retention."""

    def __init__(
        self,
        backend: Backend,
        *,
        keep_last: int = 3,
        keep_milestones_every: int = 1000,
        backpressure_minutes: float = 15.0,
    ):
        self.backend = backend
        self.keep_last = max(1, int(keep_last))
        self.keep_milestones_every = max(1, int(keep_milestones_every))
        self.backpressure_seconds = float(backpressure_minutes) * 60.0
        self.cadence = Cadence()
        self._started_mono = time.monotonic()
        self._last_success_mono: float | None = None
        self.last_success_utc: str | None = None
        self._thread = None
        self._thread_error: BaseException | None = None
        self._thread_result: dict[str, Any] | None = None
        self._lock = None
        try:
            import threading

            self._lock = threading.Lock()
        except Exception:  # pragma: no cover
            pass

    def mark_started(self) -> None:
        self._started_mono = time.monotonic()

    def _age_seconds(self) -> float:
        base = self._last_success_mono if self._last_success_mono is not None else self._started_mono
        return max(0.0, time.monotonic() - base)

    def check_backpressure(self) -> None:
        age = self._age_seconds()
        if age > self.backpressure_seconds:
            raise BackpressureError(
                f"no successful checkpoint push for {age/60:.1f} min "
                f"(limit {self.backpressure_seconds/60:.0f} min); training must pause"
            )

    def _remote_checkpoint_files(self, step: int) -> list[RemoteFile]:
        prefix = checkpoint_prefix(step) + "/"
        return self.backend.list_files(prefix)

    def verify_remote(self, step: int, manifest: CheckpointManifest) -> None:
        prefix = checkpoint_prefix(step) + "/"
        remote = {f.path[len(prefix) :]: f for f in self._remote_checkpoint_files(step)}
        expected = dict(manifest.files)
        expected[MANIFEST_NAME] = None  # local manifest size checked below
        local_manifest_size = len(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8")
        ) + 1
        if MANIFEST_NAME not in remote:
            raise CorruptCheckpointError(f"{prefix}{MANIFEST_NAME} missing remotely")
        if remote[MANIFEST_NAME].size not in (local_manifest_size, local_manifest_size - 1):
            raise CorruptCheckpointError(
                f"remote manifest size {remote[MANIFEST_NAME].size} != {local_manifest_size}"
            )
        for name, entry in expected.items():
            if name == MANIFEST_NAME:
                continue
            rf = remote.get(name)
            if rf is None:
                raise CorruptCheckpointError(f"{prefix}{name} missing remotely")
            if rf.size != entry.bytes:  # type: ignore[union-attr]
                raise CorruptCheckpointError(
                    f"{prefix}{name} size {rf.size} != {entry.bytes}"  # type: ignore[union-attr]
                )
            if rf.sha256 and rf.sha256 != entry.sha256:  # type: ignore[union-attr]
                raise CorruptCheckpointError(f"{prefix}{name} sha256 mismatch remotely")
        extra = set(remote) - set(expected)
        if extra:
            raise CorruptCheckpointError(f"{prefix} has unexpected remote files: {sorted(extra)}")
        # Re-read the commit metadata itself.  A reader must never see a
        # half-written manifest.
        remote_manifest = CheckpointManifest.from_dict(
            json.loads(self.backend.read_bytes(prefix + MANIFEST_NAME))
        )
        if remote_manifest.to_dict() != manifest.to_dict():
            raise CorruptCheckpointError("remote manifest differs from local manifest")
        remote_sums = self.backend.read_bytes(prefix + SUMS_NAME)
        if sha256_bytes(remote_sums) != manifest.files[SUMS_NAME].sha256:
            raise CorruptCheckpointError("remote SHA256SUMS sha256 mismatch")

    def publish(
        self,
        ckpt_dir: str | Path,
        *,
        step: int,
        metrics_path: str | Path | None = None,
        timeout: float = 900.0,
    ) -> dict[str, Any]:
        """Upload dir -> verify -> LATEST (separate commit) -> metrics -> retention."""
        ckpt = Path(ckpt_dir)
        manifest = read_manifest(ckpt)
        if manifest.global_step != step:
            raise FatalHubError(
                f"checkpoint dir step {manifest.global_step} != publish step {step}"
            )
        validate_checkpoint_dir(ckpt)
        t0 = time.monotonic()
        prefix = checkpoint_prefix(step)
        self.backend.upload_folder(ckpt, prefix, f"checkpoint {prefix} ({manifest.tier})")
        upload_seconds = time.monotonic() - t0
        self.verify_remote(step, manifest)
        # A separate commit: until this succeeds the previous LATEST still resolves.
        self.backend.upload_bytes(
            f"{prefix}\n".encode("utf-8"),
            LATEST,
            f"publish {prefix}",
        )
        if self.backend.read_bytes(LATEST).decode("utf-8").strip() != prefix:
            raise CorruptCheckpointError("LATEST did not round-trip after publish")
        if self.cadence.upload_seconds is None:
            self.cadence = Cadence.from_upload(upload_seconds)
        if metrics_path and Path(metrics_path).is_file():
            try:
                self.backend.upload_bytes(
                    Path(metrics_path).read_bytes(), METRICS, f"metrics at {prefix}"
                )
            except HubError:
                # Checkpoint is durable; metrics are valuable but not worth
                # failing the run over.  The next checkpoint re-uploads the file.
                pass
        self._last_success_mono = time.monotonic()
        self.last_success_utc = utcnow()
        result = {
            "step": step,
            "tier": manifest.tier,
            "upload_seconds": round(upload_seconds, 2),
            "cadence_full_seconds": round(self.cadence.full_seconds, 1),
            "cadence_weights_seconds": round(self.cadence.weights_only_seconds, 1),
        }
        try:
            self.apply_retention()
        except HubError as exc:
            result["retention_error"] = str(exc)
        return result

    # -- single-flight background publishing ------------------------------- #
    def submit(self, ckpt_dir: str | Path, *, step: int, metrics_path: str | Path | None = None) -> bool:
        import threading

        with self._lock:  # type: ignore[union-attr]
            if self._thread is not None and self._thread.is_alive():
                return False
            if self._thread is not None and self._thread_error is not None:
                # Surface a failed background publish at the next checkpoint
                # boundary instead of silently dropping it.
                err = self._thread_error
                self._thread_error = None
                raise err
            self._thread_error = None
            self._thread_result = None

            def run() -> None:
                try:
                    self._thread_result = self.publish(ckpt_dir, step=step, metrics_path=metrics_path)
                except BaseException as exc:  # propagate to the main loop
                    self._thread_error = exc

            self._thread = threading.Thread(target=run, name=f"push-{step}", daemon=True)
            self._thread.start()
            return True

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise BackpressureError("checkpoint push still running; not advancing")
        if self._thread_error is not None:
            err = self._thread_error
            self._thread_error = None
            raise err
        return self._thread_result or {}

    def push_sync(
        self,
        ckpt_dir: str | Path,
        *,
        step: int,
        metrics_path: str | Path | None = None,
    ) -> dict[str, Any]:
        self.wait()
        result = self.publish(ckpt_dir, step=step, metrics_path=metrics_path)
        self._thread_result = result
        return result

    # -- reads -------------------------------------------------------------- #
    def resolve_latest(self) -> int | None:
        try:
            text = self.backend.read_bytes(LATEST).decode("utf-8").strip()
            step = parse_checkpoint_step(text)
            if step is None:
                raise CorruptCheckpointError(f"LATEST is malformed: {text[:80]!r}")
            return step
        except (FatalHubError, CorruptCheckpointError, UnicodeDecodeError):
            steps = sorted(self.list_checkpoint_steps(), reverse=True)
            return steps[0] if steps else None

    def list_checkpoint_steps(self) -> list[int]:
        steps: set[int] = set()
        for f in self.backend.list_files(CKPT_PREFIX):
            step = parse_checkpoint_step(f.path)
            if step is not None:
                steps.add(step)
        return sorted(steps)

    def download_checkpoint(self, step: int, dest_dir: str | Path) -> CheckpointManifest:
        prefix = checkpoint_prefix(step)
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        raw = self.backend.read_bytes(f"{prefix}/{MANIFEST_NAME}")
        manifest = CheckpointManifest.from_dict(json.loads(raw))
        if manifest.global_step != step:
            raise CorruptCheckpointError(f"manifest step {manifest.global_step} != {step}")
        for rel, entry in manifest.files.items():
            target = dest / rel
            self.backend.download_to(f"{prefix}/{rel}", target)
            if target.stat().st_size != entry.bytes:
                raise CorruptCheckpointError(f"downloaded size mismatch: {rel}")
            if sha256_file(target) != entry.sha256:
                raise CorruptCheckpointError(f"downloaded sha256 mismatch: {rel}")
        # Manifest is written locally only after every payload hash verifies.
        atomic_write_bytes(dest / MANIFEST_NAME, raw)
        return manifest

    def read_manifest(self, step: int) -> CheckpointManifest:
        raw = self.backend.read_bytes(f"{checkpoint_prefix(step)}/{MANIFEST_NAME}")
        return CheckpointManifest.from_dict(json.loads(raw))

    def write_heartbeat(self, payload: dict[str, Any]) -> None:
        data = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        if len(data) > 1024:
            raise ValueError(f"heartbeat is {len(data)} bytes; contract limit is 1 KB")
        self.backend.upload_bytes(data, HEARTBEAT, "heartbeat")

    def apply_retention(self) -> list[int]:
        steps = self.list_checkpoint_steps()
        if not steps:
            return []
        latest = self.resolve_latest()
        keep: set[int] = set()
        if latest is not None:
            keep.add(latest)
        keep.update(steps[-self.keep_last :])
        for step in steps:
            if step % self.keep_milestones_every == 0:
                keep.add(step)
        # Keep the newest full checkpoints separately: a weights_only checkpoint
        # must never be the only survivor.
        full_steps: list[int] = []
        for step in reversed(steps):
            try:
                m = self.read_manifest(step)
            except HubError:
                full_steps.append(step)  # unknown -> conservatively keep
                continue
            if m.tier == "full":
                full_steps.append(step)
            if len(full_steps) >= self.keep_last:
                break
        keep.update(full_steps)
        delete = [s for s in steps if s not in keep]
        if not delete:
            return []
        paths = [
            f.path
            for f in self.backend.list_files(CKPT_PREFIX)
            if parse_checkpoint_step(f.path) in set(delete)
        ]
        self.backend.delete_paths(paths, f"retention: delete {delete}")
        return delete

    def status(self) -> dict[str, Any]:
        latest = self.resolve_latest()
        steps = self.list_checkpoint_steps()
        out: dict[str, Any] = {
            "repo": getattr(self.backend, "repo_id", str(getattr(self.backend, "root", "local"))),
            "latest": latest,
            "checkpoints": steps,
            "last_push_utc": self.last_success_utc,
        }
        if latest is not None:
            try:
                m = self.read_manifest(latest)
                out.update(
                    {
                        "run_id": m.run_id,
                        "global_step": m.global_step,
                        "consumed_tokens": m.consumed_tokens,
                        "loss_ema": m.loss_ema,
                        "tier": m.tier,
                        "cursor": m.cursor,
                        "created_utc": m.created_utc,
                    }
                )
            except HubError as exc:
                out["manifest_error"] = str(exc)
        try:
            hb = json.loads(self.backend.read_bytes(HEARTBEAT))
            out["heartbeat"] = hb
        except (HubError, json.JSONDecodeError, UnicodeDecodeError):
            pass
        try:
            out["repo_bytes"] = self.backend.repo_size()
        except HubError:
            pass
        return out


# --------------------------------------------------------------------------- #
# Upload throughput probe
# --------------------------------------------------------------------------- #
def probe_upload(backend: Backend, size_bytes: int = 2 * 1024 * 1024) -> float:
    """Upload random bytes once and return seconds.  Leaves no probe file."""
    data = os.urandom(size_bytes)
    path = f"_probe/{uuid.uuid4().hex}.bin"
    t0 = time.monotonic()
    backend.upload_bytes(data, path, "bandwidth probe")
    seconds = time.monotonic() - t0
    try:
        backend.delete_paths([path], "remove bandwidth probe")
    except HubError:
        pass
    return seconds
