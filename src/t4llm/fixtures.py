"""Tiny random model + synthetic shards for CPU/GPU resilience tests only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import atomic_write_text


def make_tiny_model(dest: str | Path, *, seq_len: int = 256, vocab_size: int = 128) -> Path:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=max(seq_len, 256),
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(str(out), safe_serialization=True)

    vocab = {f"tok{i}": i for i in range(vocab_size)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="tok1",
        pad_token="tok0",
        unk_token="tok0",
    )
    fast.save_pretrained(str(out))
    # A tiny chat template for SFT smoke tests; never used by CPT.
    atomic_write_text(
        out / "chat_template.jinja",
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}",
    )
    return out


def make_tiny_run(
    root: str | Path,
    *,
    seq_len: int = 64,
    num_samples: int = 4000,
    max_steps: int = 200,
    ckpt_steps: int = 10,
    micro_batch: int = 1,
    grad_accum: int = 1,
    lr: float = 1e-3,
    seed: int = 3407,
    max_samples_per_shard: int = 1000,
) -> dict[str, Any]:
    from .data import build_synthetic_shards

    r = Path(root)
    r.mkdir(parents=True, exist_ok=True)
    model_dir = make_tiny_model(r / "tiny-model", seq_len=seq_len)
    data_dir = r / "data"
    index = build_synthetic_shards(
        data_dir,
        vocab_size=128,
        seq_len=seq_len,
        num_samples=num_samples,
        seed=seed,
        max_samples_per_shard=max_samples_per_shard,
    )
    cfg = {
        "run": {
            "base_slug": "tiny",
            "phase": "cpt",
            "hub_repo": None,
            "data_repo": None,
            "data_revision": index["data_revision"],
            "seed": seed,
        },
        "model": {
            "base": str(model_dir),
            "load_in_4bit": False,
            "attn": "sdpa",
            "gradient_checkpointing": "unsloth",
            "trust_remote_code": False,
        },
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "use_rslora": True,
            "target": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "train": {
            "dtype": "fp16",
            "seq_len": seq_len,
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "lr": lr,
            "scheduler": "cosine",
            "warmup_ratio": 0.02,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "optim": "paged_adamw_8bit",
            "max_steps": max_steps,
            "logging_steps": 1,
        },
        "resilience": {
            "ckpt_every_minutes": 6,
            "ckpt_every_steps": ckpt_steps,
            "heartbeat_seconds": 120,
            "time_budget_minutes": 5,
            "keep_last": 3,
            "keep_milestones_every": 1000,
            "push_backpressure_minutes": 15,
            "resume_loss_tolerance": 0.05,
            "max_rollbacks": 3,
        },
        "data": {
            "revision": index["data_revision"],
            "seq_len": seq_len,
            "local_dir": str(data_dir),
            "source_manifest": None,
            "max_tokens_per_shard": 5_000_000,
        },
        "eval": {"domain_set": None, "general_samples": 512, "seed": seed},
    }
    config_path = r / "config.yaml"
    atomic_write_text(config_path, json.dumps(cfg, indent=2) + "\n")
    return {
        "root": str(r),
        "config": str(config_path),
        "model": str(model_dir),
        "data": str(data_dir),
        "data_revision": index["data_revision"],
        "num_samples": index["total_samples"],
    }
