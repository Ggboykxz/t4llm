"""Model/LoRA construction with the fp16 + SDPA policy enforced (§1, §5, §11)."""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .state import ConfigError, _params_b


class ModelError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    model: Any
    notes: list[str] = field(default_factory=list)
    params_b: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    # Turing has no FlashAttention; make the mem-efficient SDPA backend explicit.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass


def build_model(
    cfg: dict[str, Any], *, device: str | None = None, attach_lora: bool = True
) -> ModelBundle:
    from transformers import AutoModelForCausalLM

    mcfg = cfg["model"]
    base = mcfg["base"]
    on_cuda = (device or ("cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda")
    if not on_cuda and os.environ.get("T4LLM_ALLOW_NON_T4") != "1":
        raise ModelError("CPU model build requires T4LLM_ALLOW_NON_T4=1 (tests only)")
    dtype = torch.float16 if on_cuda else torch.float32
    attn = mcfg.get("attn", "sdpa")
    if "flash" in (attn or "").lower():
        raise ModelError("FlashAttention is forbidden on sm_75")
    params_b = _params_b(base) or _params_b(mcfg.get("base_slug", ""))
    notes: list[str] = []

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": attn,
        "trust_remote_code": bool(mcfg.get("trust_remote_code", False)),
        "low_cpu_mem_usage": True,
    }
    if mcfg.get("load_in_4bit"):
        if params_b and params_b < 4.0 and os.environ.get("T4LLM_ALLOW_SMALL_4BIT") != "1":
            raise ModelError("4-bit is only permitted for >=4B bases (or logged OOM escalation)")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("torch_dtype", None)
        notes.append("4-bit NF4 enabled; LoRA fp16")
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model.config.use_cache = False
    if on_cuda:
        model = model.to("cuda")

    if not mcfg.get("load_in_4bit"):
        # fp16 storage is mandatory; reject a silently-upcast/bf16 model.
        if model.dtype == torch.bfloat16:
            raise ModelError("model loaded in bf16; this is forbidden")
        model = model.half() if on_cuda else model.float()

    gc = mcfg.get("gradient_checkpointing", "unsloth")
    if gc:
        if gc == "unsloth":
            try:
                import unsloth  # noqa: F401

                notes.append("unsloth present; using standard HF checkpointing wrapper for portability")
            except Exception:
                notes.append("unsloth unavailable; using standard non-reentrant HF checkpointing")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if attach_lora:
        lcfg = cfg["lora"]
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lcfg["r"]),
            lora_alpha=int(lcfg["alpha"]),
            lora_dropout=float(lcfg.get("dropout", 0.0)),
            target_modules=list(lcfg["target"]),
            bias="none",
            use_rslora=bool(lcfg.get("use_rslora", False)),
        )
        try:
            model = get_peft_model(model, lora)
        except Exception as exc:
            raise ModelError(f"LoRA target modules rejected: {exc}") from exc
    cast_trainable_fp32(model)
    model.train()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    if on_cuda:
        torch.cuda.empty_cache()
    return ModelBundle(model=model, notes=notes, params_b=params_b)


def cast_trainable_fp32(model: Any) -> None:
    """Keep frozen base in fp16; trainable LoRA adapters need fp32 master weights.

    GradScaler cannot unscale fp16 gradients.  fp32 adapters + fp16 base is the
    standard T4-compatible form of fp16 LoRA (ADR-0002).
    """
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()


def save_adapter(model: Any, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    # The checkpoint contract says exactly the listed files; PEFT occasionally
    # writes a README.  Remove anything that is not part of the adapter.
    for p in out.iterdir():
        if p.is_file() and p.name not in {"adapter_model.safetensors", "adapter_config.json"}:
            p.unlink()
    if (out / "adapter_model.bin").exists():
        raise ModelError("PEFT emitted an unsafe pickle adapter; expected safetensors")


def load_adapter(model: Any, adapter_dir: str | Path) -> Any:
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)


def build_optimizer(model: Any, cfg: dict[str, Any], *, allow_fallback: bool = False) -> tuple[torch.optim.Optimizer, list[str]]:
    tcfg = cfg["train"]
    notes: list[str] = []
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ModelError("no trainable parameters; LoRA did not attach")
    name = str(tcfg.get("optim", "paged_adamw_8bit"))
    total_params = sum(p.numel() for p in model.parameters())
    if not torch.cuda.is_available() or total_params < 10_000_000:
        notes.append(
            f"paged_adamw_8bit not applicable on this test/CPU model ({total_params:,} params); "
            "using adamw_torch"
        )
        name = "adamw_torch"
    if name == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.PagedAdamW8bit(
                params,
                lr=float(tcfg["lr"]),
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=float(tcfg.get("weight_decay", 0.0)),
            )
            return opt, notes
        except Exception as exc:
            msg = f"paged_adamw_8bit unavailable ({type(exc).__name__}: {exc})"
            if not allow_fallback:
                raise ModelError(
                    msg + "; install bitsandbytes or explicitly allow the fp32 fallback"
                ) from exc
            notes.append(msg + "; using adamw_torch (recorded in state)")
    opt = torch.optim.AdamW(
        params,
        lr=float(tcfg["lr"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        foreach=False,
    )
    return opt, notes


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any], total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    tcfg = cfg["train"]
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.0))
    warmup = max(0, int(total_steps * warmup_ratio))
    max_steps = max(1, int(total_steps))
    min_ratio = 0.0
    kind = str(tcfg.get("scheduler", "cosine")).lower()
    if kind != "cosine":
        raise ConfigError("only cosine scheduler is supported")

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: str = "cuda") -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def assert_no_bf16(model: Any) -> None:
    for name, p in model.named_parameters():
        if p.dtype == torch.bfloat16:
            raise ModelError(f"bf16 parameter detected: {name}")
