"""GPU probe, dtype policy and hard capability asserts (§1, §4).

Nothing in this module imports torch at module scope: importing t4llm sets the
required environment variables before any CUDA context exists.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MIN_CC = (7, 0)
MIN_VRAM_GIB = 14.0
FORBIDDEN_ATTN = {"flash_attention_2", "flash_attention_3", "fa2", "fa3", "flash"}
ALLOWED_ATTN = {"sdpa", "eager"}


class EnvError(RuntimeError):
    """Unsupported runtime.  Never caught to continue training silently."""


@dataclass(frozen=True)
class EnvReport:
    python: str
    platform: str
    torch_version: str
    cuda_version: str | None
    cuda_available: bool
    gpu_name: str
    cc_major: int
    cc_minor: int
    compute_capability: str
    vram_gib: float
    bf16_reported: bool
    sdpa: bool
    git_sha: str
    training_capable: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def configure_process_env() -> None:
    """Idempotent.  Safe to call from every entry point."""
    defaults = {
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "offline",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def git_sha() -> str:
    explicit = os.environ.get("T4LLM_GIT_SHA")
    if explicit:
        return explicit
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def probe() -> EnvReport:
    """Best-effort probe.  Does not raise; use assert_capable() to gate work."""
    configure_process_env()
    import torch

    cuda = bool(torch.cuda.is_available())
    name = "none"
    major = minor = 0
    vram = 0.0
    bf16 = False
    sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    if cuda:
        props = torch.cuda.get_device_properties(0)
        name = props.name
        major, minor = torch.cuda.get_device_capability(0)
        vram = round(props.total_memory / 2**30, 2)
        try:
            bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16 = False

    reasons: list[str] = []
    if not cuda:
        reasons.append("CUDA is not available")
    if cuda and (major, minor) < MIN_CC:
        reasons.append(
            f"compute capability {major}.{minor} < required {MIN_CC[0]}.{MIN_CC[1]}"
        )
    if cuda and vram < MIN_VRAM_GIB:
        reasons.append(f"VRAM {vram:.2f} GiB < required {MIN_VRAM_GIB:.1f} GiB")
    if not sdpa:
        reasons.append("torch.nn.functional.scaled_dot_product_attention missing")

    return EnvReport(
        python=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cuda_available=cuda,
        gpu_name=name,
        cc_major=major,
        cc_minor=minor,
        compute_capability=f"{major}.{minor}",
        vram_gib=vram,
        bf16_reported=bf16,
        sdpa=sdpa,
        git_sha=git_sha(),
        training_capable=not reasons,
        reasons=tuple(reasons),
    )


def assert_capable(
    report: EnvReport,
    *,
    dtype: str = "fp16",
    attn: str = "sdpa",
    load_in_4bit: bool = False,
    model_params_b: float | None = None,
    allow_non_t4: bool = False,
) -> None:
    """Hard-fail with every violation listed, not just the first."""
    errs = list(report.reasons)
    # Derive hardware violations from the report fields as well as the cached
    # reasons: callers may construct a report directly (tests, CI fixtures) and
    # a stale/empty reasons tuple must never wave a bad device through.
    if not report.cuda_available:
        errs.append("CUDA is not available")
    if report.cuda_available and (report.cc_major, report.cc_minor) < MIN_CC:
        errs.append(
            f"compute capability {report.cc_major}.{report.cc_minor} "
            f"< required {MIN_CC[0]}.{MIN_CC[1]}"
        )
    if report.cuda_available and report.vram_gib < MIN_VRAM_GIB:
        errs.append(f"VRAM {report.vram_gib:.2f} GiB < required {MIN_VRAM_GIB:.1f} GiB")
    if not report.sdpa:
        errs.append("torch.nn.functional.scaled_dot_product_attention missing")
    dt = (dtype or "").lower()
    at = (attn or "").lower()

    if dt not in {"fp16", "float16"}:
        errs.append(f"dtype={dtype!r}; only fp16 is permitted on this project")
    if at in FORBIDDEN_ATTN or "flash" in at:
        errs.append(
            f"attn={attn!r} requires sm_80+; use 'sdpa' (mem-efficient backend) or 'eager'"
        )
    elif at not in ALLOWED_ATTN:
        errs.append(f"attn={attn!r}; allowed: {sorted(ALLOWED_ATTN)}")
    if load_in_4bit and model_params_b is not None and model_params_b < 4.0:
        errs.append(
            f"load_in_4bit=True for a {model_params_b:.1f}B model; fp16 LoRA is required "
            "for models < 4B (an explicit OOM escalation must be recorded in state)"
        )
    if allow_non_t4:
        # CI/CPU tests may run fp32 tiny models; hardware-only reasons are waived,
        # but dtype/attention policy violations above are never waived.
        errs = [
            e
            for e in errs
            if not (
                e == "CUDA is not available"
                or e.startswith("compute capability")
                or e.startswith("VRAM ")
            )
        ]
    if errs:
        raise EnvError("; ".join(dict.fromkeys(errs)))


def require_t4(
    *,
    dtype: str = "fp16",
    attn: str = "sdpa",
    load_in_4bit: bool = False,
    model_params_b: float | None = None,
) -> EnvReport:
    report = probe()
    allow = os.environ.get("T4LLM_ALLOW_NON_T4") == "1"
    assert_capable(
        report,
        dtype=dtype,
        attn=attn,
        load_in_4bit=load_in_4bit,
        model_params_b=model_params_b,
        allow_non_t4=allow,
    )
    return report


def capability_report() -> dict[str, Any]:
    return probe().to_dict()


def print_capability_report() -> EnvReport:
    r = probe()
    print("T4LLM capability report")
    print(f"  python            : {r.python}")
    print(f"  platform          : {r.platform}")
    print(f"  torch             : {r.torch_version} (cuda {r.cuda_version})")
    print(f"  gpu               : {r.gpu_name}")
    print(f"  compute capability: {r.compute_capability} (need >= 7.0)")
    print(f"  vram              : {r.vram_gib:.2f} GiB (need >= 14.0)")
    print(f"  sdpa              : {r.sdpa}")
    print(f"  bf16 reported     : {r.bf16_reported} (ignored: policy is fp16 only)")
    print(f"  git sha           : {r.git_sha}")
    if r.training_capable:
        print("  verdict           : TRAINING-CAPABLE")
    else:
        print("  verdict           : NOT TRAINING-CAPABLE")
        for reason in r.reasons:
            print(f"    - {reason}")
    return r


def write_env_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = capability_report()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, p)
    return data
