"""Supervised fine-tuning: chat-template prompt masking, no packing.

The durability machinery lives in :mod:`t4llm.phase_core`; this module owns the
SFT-specific correctness rules:

* every example keeps its own sequence (no cross-example packing);
* prompt tokens get label ``-100`` and are never trained on;
* the completion (including the template's assistant terminator) is supervised;
* truncation keeps at least one completion token by dropping prompt tokens from
  the left, never by silently dropping the whole example.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .phase_core import PhaseBatcher, PhaseDataset, train_phase
from .state import ConfigError


class SFTDataset(PhaseDataset):
    phase = "sft"

    def validate_record(self, rec: dict[str, Any], lineno: int) -> None:
        super().validate_record(rec, lineno)
        if "messages" in rec:
            messages = rec["messages"]
            if not isinstance(messages, list) or not messages:
                raise ConfigError(f"{self.path}:{lineno}: messages must be a non-empty list")
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    raise ConfigError(f"{self.path}:{lineno}: message {i} must be an object")
                if msg.get("role") not in {"system", "user", "assistant", "tool"}:
                    raise ConfigError(
                        f"{self.path}:{lineno}: message {i} has invalid role "
                        f"{msg.get('role')!r}"
                    )
                if not isinstance(msg.get("content"), str):
                    raise ConfigError(
                        f"{self.path}:{lineno}: message {i} needs string content"
                    )
            if messages[-1].get("role") != "assistant":
                raise ConfigError(
                    f"{self.path}:{lineno}: last message must be the assistant turn "
                    "to be supervised"
                )
            if not messages[-1]["content"].strip():
                raise ConfigError(f"{self.path}:{lineno}: assistant turn is empty")
            return
        if "prompt" not in rec or "response" not in rec:
            raise ConfigError(
                f"{self.path}:{lineno}: need 'messages' or 'prompt'+'response'"
            )
        if not isinstance(rec["prompt"], str):
            raise ConfigError(f"{self.path}:{lineno}: prompt must be a string")
        if not isinstance(rec["response"], str) or not rec["response"].strip():
            raise ConfigError(f"{self.path}:{lineno}: response must be a non-empty string")

    def make_batcher(self, tokenizer: Any) -> "SFTBatcher":
        return SFTBatcher(self, tokenizer, self.seq_len)


class SFTBatcher(PhaseBatcher):
    def encode(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        if "messages" in rec:
            messages = [
                {"role": str(m["role"]), "content": str(m["content"])}
                for m in rec["messages"]
            ]
            if not getattr(self.tokenizer, "chat_template", None):
                raise ConfigError(
                    "SFT record uses messages but the tokenizer has no chat template; "
                    "use prompt/response records or supply a tokenizer with a template"
                )
            prompt_ids = self._chat_ids(messages[:-1], generation=True)
            full_ids = self._chat_ids(messages, generation=False)
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ConfigError(
                    "chat template cannot be split deterministically into prompt "
                    "and assistant completion; refusing to guess"
                )
            completion_ids = full_ids[len(prompt_ids) :]
        else:
            pair = self._prompt_completion(str(rec["prompt"]), str(rec["response"]))
            if pair is None:
                return None
            prompt_ids, completion_ids = pair
        truncated = self._truncate(prompt_ids, completion_ids, self.seq_len)
        if truncated is None:
            return None
        ids, labels = truncated
        supervised = sum(1 for x in labels if x != -100)
        if supervised == 0:
            return None
        return {"ids": ids, "labels": labels, "supervised_tokens": supervised}

    def collate(self, encoded: list[dict[str, Any]]) -> dict[str, Any]:
        if not encoded:
            raise ConfigError("SFT collate called with no examples")
        return self._pad([(x["ids"], x["labels"]) for x in encoded])


def sft_loss(
    model: Any,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    *,
    on_cuda: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    alpha = float(cfg["train"].get("neftune_noise_alpha") or 0.0)
    if alpha > 0.0:
        # NEFTune: uniform embedding noise scaled by alpha/sqrt(L*d).  Applied
        # only to the training forward, never to evaluation or reference passes.
        embeds = model.get_input_embeddings()(batch["input_ids"])
        if on_cuda:
            noise = torch.empty_like(embeds).uniform_(-1.0, 1.0)
            scale = alpha / math.sqrt(max(1, embeds.size(1) * embeds.size(2)))
            embeds = embeds + scale * noise
        out = model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)
    else:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=attention_mask,
            labels=labels,
        )
    if out.loss is None:
        raise ConfigError("SFT model returned no loss; check the labels")
    return out.loss, {}


def train_sft(
    config_path: str | Path,
    *,
    time_budget: str | int | float | None = None,
    max_steps: int | None = None,
    run_id: str | None = None,
    allow_config_change: bool = False,
    train_file: str | Path | None = None,
    model: str | None = None,
    device: str | None = None,
) -> int:
    # Load once here (cheap) to reject a misconfigured phase before building a
    # model.  train_phase reloads with the same deterministic identity.
    from .phase_core import load_phase_config

    prepared = load_phase_config(config_path, train_file=train_file, model=model)
    train = prepared.cfg["train"]
    if bool(train.get("packing", False)):
        raise ConfigError("SFT must not pack; set train.packing: false")
    if not bool(train.get("train_on_responses_only", True)):
        raise ConfigError(
            "SFT would train on prompt tokens; set train.train_on_responses_only: true "
            "(the phase trainer never silently changes this)"
        )
    return train_phase(
        "sft",
        config_path,
        time_budget=time_budget,
        max_steps=max_steps,
        run_id=run_id,
        allow_config_change=allow_config_change,
        train_file=train_file,
        model=model,
        device=device,
    )
