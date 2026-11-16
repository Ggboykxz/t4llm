"""GGUF conversion guards: no converter, wrong magic, and a clean GGUF write."""
from __future__ import annotations

from pathlib import Path

import pytest

from t4llm import export


def _fake_converter(root: Path, payload: bytes) -> Path:
    script = root / "convert_hf_to_gguf.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"out = sys.argv[sys.argv.index('--outfile') + 1]\n"
        f"Path(out).write_bytes({payload!r})\n",
        encoding="utf-8",
    )
    return script


def test_convert_to_gguf_requires_a_converter(monkeypatch, tmp_path):
    def missing() -> Path:
        raise export.ExportError("llama.cpp converter missing")

    monkeypatch.setattr(export, "_llama_cpp_dir", missing)
    with pytest.raises(export.ExportError, match="converter missing"):
        export.convert_to_gguf(tmp_path, tmp_path / "model-f16.gguf")


def test_convert_to_gguf_rejects_a_non_gguf_file(monkeypatch, tmp_path):
    _fake_converter(tmp_path, b"NOPE")
    monkeypatch.setattr(export, "_llama_cpp_dir", lambda: tmp_path)
    with pytest.raises(export.ExportError, match="not a GGUF"):
        export.convert_to_gguf(tmp_path, tmp_path / "model-f16.gguf")


def test_convert_to_gguf_accepts_a_gguf_magic(monkeypatch, tmp_path):
    _fake_converter(tmp_path, b"GGUF" + b"\x00" * 32)
    monkeypatch.setattr(export, "_llama_cpp_dir", lambda: tmp_path)
    out = export.convert_to_gguf(tmp_path, tmp_path / "model-f16.gguf")
    assert out.read_bytes()[:4] == b"GGUF"
