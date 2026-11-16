"""Static checks for the run notebook: valid JSON, no secrets, compilable cells."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "run.ipynb"

SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
)

# IPython escapes that are not Python and must be dropped before compiling.
MAGIC_PREFIXES = ("!", "%")


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def test_notebook_is_valid_nbformat4():
    nb = _load()
    assert nb["nbformat"] == 4
    assert isinstance(nb["nbformat_minor"], int)
    assert nb["cells"], "the notebook must not be empty"
    assert nb["metadata"]["kernelspec"]["language"] == "python"


def test_notebook_contains_no_token_literals():
    raw = "\n".join(
        [_source(cell) for cell in _load()["cells"]]
        + [json.dumps(cell.get("outputs", [])) for cell in _load()["cells"]]
    )
    for pattern in SECRET_PATTERNS:
        assert not pattern.search(raw), f"token-like literal in notebook: {pattern.pattern}"


def test_notebook_code_cells_have_no_saved_output():
    for index, cell in enumerate(_load()["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None, f"cell {index} has an execution count"
        assert cell.get("outputs") == [], f"cell {index} has saved output (may contain secrets)"


@pytest.mark.parametrize(
    "path",
    [
        "scripts/bootstrap.sh",
        "scripts/push_hf_backup.py",
        "configs/cpt.yaml",
        "configs/sft.yaml",
        "configs/dpo.yaml",
    ],
)
def test_notebook_referenced_files_exist(path):
    nb = _load()
    text = "\n".join(_source(cell) for cell in nb["cells"])
    assert path in text, f"notebook no longer references {path}"
    root = NOTEBOOK.parents[1]
    assert (root / path).exists(), f"notebook references missing {path}"


def test_notebook_code_cells_compile_as_python():
    for index, cell in enumerate(_load()["cells"]):
        if cell.get("cell_type") != "code":
            continue
        lines = []
        for line in _source(cell).splitlines():
            if line.lstrip().startswith(MAGIC_PREFIXES):
                lines.append("")  # drop shell/line-magics, keep line numbers
            else:
                lines.append(line)
        source = "\n".join(lines)
        try:
            compile(source, f"<notebook cell {index}>", "exec")
        except SyntaxError as exc:  # pragma: no cover - failure path
            raise AssertionError(f"cell {index} does not compile: {exc}") from exc
