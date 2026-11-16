from __future__ import annotations

import re
from pathlib import Path

import pytest

from t4llm.env import EnvError, EnvReport, assert_capable


def report(**overrides):
    base = dict(
        python="3.13",
        platform="test",
        torch_version="2.14",
        cuda_version="13.0",
        cuda_available=True,
        gpu_name="Tesla T4",
        cc_major=7,
        cc_minor=5,
        compute_capability="7.5",
        vram_gib=14.56,
        bf16_reported=True,
        sdpa=True,
        git_sha="x",
        training_capable=True,
        reasons=(),
    )
    base.update(overrides)
    return EnvReport(**base)


def test_valid_t4_passes():
    assert_capable(report())


@pytest.mark.parametrize(
    "bad, needle",
    [
        (report(cc_major=6, cc_minor=1), "compute capability"),
        (report(vram_gib=12.0), "VRAM"),
        (report(sdpa=False), "scaled_dot_product"),
    ],
)
def test_hardware_gate(bad, needle):
    with pytest.raises(EnvError, match=needle):
        assert_capable(bad)


def test_bf16_rejected():
    with pytest.raises(EnvError, match="dtype"):
        assert_capable(report(), dtype="bf16")


def test_flash_attention_rejected():
    with pytest.raises(EnvError, match="sm_80"):
        assert_capable(report(), attn="flash_attention_2")


def test_small_model_4bit_rejected():
    with pytest.raises(EnvError, match="fp16 LoRA"):
        assert_capable(report(), load_in_4bit=True, model_params_b=1.7)


def test_static_policy():
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        re.compile(r"torch_dtype\s*=\s*torch\.bfloat16"),
        re.compile(r"dtype\s*=\s*['\"]bf16", re.I),
        re.compile(r"attn_implementation\s*=\s*['\"]flash_attention_2"),
        re.compile(r"bf16\s*=\s*True"),
    ]
    for p in list((root / "src").rglob("*.py")) + list((root / "configs").glob("*.yaml")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in forbidden:
            assert not pat.search(text), f"forbidden runtime pattern {pat.pattern} in {p}"
    for p in (root / "configs").glob("*.yaml"):
        text = p.read_text()
        if "dtype:" in text:
            assert "dtype: fp16" in text
        if "attn:" in text:
            assert 'attn: "sdpa"' in text or "attn: sdpa" in text
