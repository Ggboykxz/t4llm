#!/usr/bin/env python
"""Push a self-contained backup of this repo to a private HF Hub repo.

The code is the only durable artifact we can publish before the domain, data
licences and final base model are confirmed.  Training checkpoints are *not*
uploaded here: they already live in the run repo through the normal checkpoint
path.  This script exists so a dying notebook session cannot lose the source.

Usage:

    export HF_TOKEN=...                 # never write it to a file
    python scripts/push_hf_backup.py --repo Ggboykxz/t4llm-pipeline \
        --create-private

Uploads, under ``backup/`` in the target repo:

* ``t4llm-<commit>.tar.gz``  — ``git archive`` of HEAD (source only)
* ``t4llm-<commit>.bundle``  — full git history for an exact restore
* ``LATEST.json``            — commit, creation time, sizes, SHA-256 digests

Refuses to run when the worktree is dirty, so the backup always corresponds to
a real commit.  Pass ``--allow-dirty`` only for a throwaway snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> bytes:
    return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.PIPE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return _run(["git", *args], root).decode("utf-8").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="<owner>/<name> on the Hub")
    parser.add_argument("--root", default=".", help="git worktree to archive")
    parser.add_argument("--create-private", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="build locally, do not upload")
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("HF_TOKEN is not set; refusing to upload", file=sys.stderr)
        return 2
    if "/" not in args.repo:
        print("--repo must look like <owner>/<name>", file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    status = _git(root, "status", "--porcelain")
    if status and not args.allow_dirty:
        print("worktree is dirty; commit first (or pass --allow-dirty):", file=sys.stderr)
        print(status, file=sys.stderr)
        return 2
    commit = _git(root, "rev-parse", "HEAD")
    short = _git(root, "rev-parse", "--short=12", "HEAD")

    work = Path(tempfile.mkdtemp(prefix="t4llm-backup-"))
    archive = work / f"t4llm-{short}.tar.gz"
    bundle = work / f"t4llm-{short}.bundle"
    _run(["git", "archive", "--format=tar.gz", f"--prefix=t4llm-{short}/", "-o", str(archive), "HEAD"], root)
    _run(["git", "bundle", "create", str(bundle), "--all"], root)

    meta = {
        "commit": commit,
        "short": short,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": _sha256(archive)},
        "bundle": {"name": bundle.name, "bytes": bundle.stat().st_size, "sha256": _sha256(bundle)},
        "note": "Source backup only; training checkpoints live in the run repo.",
    }
    (work / "LATEST.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))

    if args.dry_run:
        print(f"dry run: artifacts in {work}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        args.repo,
        repo_type="model",
        private=True,
        exist_ok=True,
    ) if args.create_private else None
    try:
        api.repo_info(args.repo, repo_type="model")
    except Exception as exc:
        print(f"repo {args.repo!r} is not accessible: {type(exc).__name__}", file=sys.stderr)
        return 3

    for path in (archive, bundle):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"backup/{path.name}",
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"backup source {short}",
        )
    api.upload_file(
        path_or_fileobj=str(work / "LATEST.json"),
        path_in_repo="backup/LATEST.json",
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"point backup/LATEST.json at {short}",
    )
    print(f"pushed {short} to https://huggingface.co/{args.repo}/tree/main/backup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
