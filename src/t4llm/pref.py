"""Direct preference optimisation with a frozen reference policy.

Reference log-probabilities are computed by disabling the LoRA adapter on the
same model (``disable_adapter``), so DPO needs one 1.7B model in VRAM rather
than two.  The reference pass is ``no_grad`` and its logits are freed before the
policy forward, which keeps the peak below the T4 budget at ``micro_batch=1``.

Supported loss is the standard sigmoid DPO objective (``dpo``/``sigmoid``).
ORPO/IPO and friends are refused loudly rather than approximated: a different
objective changes the optimisation trajectory and therefore the run identity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .phase_core import PhaseBatcher, PhaseDataset, train_phase
from .state import ConfigError


class DPODataset(PhaseDataset):
    phase = "dpo"

    def validate_record(self, rec: dict[str, Any], lineno: int) -> None:
        super().validate_record(rec, lineno)
        for key in ("prompt", "chosen", "rejected"):
            if key not in rec or not isinstance(rec[key], str):
                raise ConfigError(f"{self.path}:{lineno}: {key!r} must be a string")
        if not rec["chosen"].strip():
            raise ConfigError(f"{self.path}:{lineno}: chosen completion is empty")
        if not rec["rejected"].strip():
            raise ConfigError(f"{self.path}:{lineno}: rejected completion is empty")
        if rec.get("system") is not None and not isinstance(rec["system"], str):
            raise ConfigError(f"{self.path}:{lineno}: system must be a string when present")

    def make_batcher(self, tokenizer: Any) -> "DPOBatcher":
        return DPOBatcher(self, tokenizer, self.seq_len)


class DPOBatcher(PhaseBatcher):
    def encode(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        system = rec.get("system")
        chosen = self._prompt_completion(
            str(rec["prompt"]), str(rec["chosen"]), system=system
        )
        rejected = self._prompt_completion(
            str(rec["prompt"]), str(rec["rejected"]), system=system
        )
        if chosen is None or rejected is None:
            return None
        c = self._truncate(chosen[0], chosen[1], self.seq_len)
        r = self._truncate(rejected[0], rejected[1], self.seq_len)
        if c is None or r is None:
            return None
        supervised = sum(1 for x in c[1] if x != -100) + sum(
            1 for x in r[1] if x != -100
        )
        if supervised < 2:
            return None
        return {
            "chosen": c,
            "rejected": r,
            "supervised_tokens": supervised,
        }

    def collate(self, encoded: list[dict[str, Any]]) -> dict[str, Any]:
        if not encoded:
            raise ConfigError("DPO collate called with no pairs")
        # Keep pairs aligned: all chosen rows, then all rejected rows.
        rows = [x["chosen"] for x in encoded] + [x["rejected"] for x in encoded]
        batch = self._pad(rows)
        batch["pairs"] = len(encoded)
        return batch


def _gather_logprobs(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-sequence summed log-probability of the supervised tokens.

    ``logits`` is fp16 ``[B,T,V]``.  The log-sum-exp is accumulated one position
    chunk at a time in fp32 so a 151k vocabulary does not require an fp32 copy of
    the full tensor.
    """
    bsz, steps, _ = logits.shape
    lse = torch.empty((bsz, steps), dtype=torch.float32, device=logits.device)
    chunk = 256
    for start in range(0, steps, chunk):
        end = min(steps, start + chunk)
        lse[:, start:end] = torch.logsumexp(
            logits[:, start:end].float(), dim=-1
        )
    target = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1).float()
    token_logp = (target - lse) * mask.float()
    return token_logp.sum(dim=-1)


def dpo_loss(
    model: Any,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    *,
    on_cuda: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    beta = float(cfg["train"].get("beta", 0.1))
    if beta <= 0:
        raise ConfigError("train.beta must be positive for DPO")
    loss_type = str(cfg["train"].get("loss_type", "sigmoid")).lower()
    if loss_type not in {"dpo", "sigmoid"}:
        raise ConfigError(
            f"loss_type={loss_type!r} is not implemented; this trainer ships the "
            "standard sigmoid DPO objective only"
        )
    if not hasattr(model, "disable_adapter"):
        raise ConfigError(
            "DPO reference policy requires a PEFT model with disable_adapter(); "
            "refusing to duplicate or silently skip the reference"
        )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    targets = input_ids[:, 1:]
    mask = labels[:, 1:] != -100
    if not bool(mask.any()):
        raise ConfigError("DPO batch has no supervised completion tokens")

    # Reference first, with no graph, then free its logits before the policy
    # forward.  At seq_len=2048 this saves roughly one full logits tensor.
    with torch.no_grad():
        with model.disable_adapter():
            ref_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits[:, :-1]
            ref_sums = _gather_logprobs(ref_logits, targets, mask)
        del ref_logits

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    policy_sums = _gather_logprobs(logits, targets, mask)
    del logits

    pairs = int(batch["pairs"])
    if policy_sums.numel() != 2 * pairs:
        raise ConfigError(
            f"DPO batch shape mismatch: {policy_sums.numel()} rows for {pairs} pairs"
        )
    pi_c, pi_r = policy_sums[:pairs], policy_sums[pairs:]
    ref_c, ref_r = ref_sums[:pairs], ref_sums[pairs:]
    logits_diff = (pi_c - pi_r) - (ref_c - ref_r)
    losses = -F.logsigmoid(beta * logits_diff)
    chosen_reward = beta * (pi_c - ref_c)
    rejected_reward = beta * (pi_r - ref_r)
    metrics = {
        "reward_margin": float((chosen_reward - rejected_reward).mean().detach().cpu()),
        "chosen_reward": float(chosen_reward.mean().detach().cpu()),
        "rejected_reward": float(rejected_reward.mean().detach().cpu()),
        "dpo_accuracy": float((logits_diff > 0).float().mean().detach().cpu()),
    }
    return losses.mean(), metrics


def train_dpo(
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
    from .phase_core import load_phase_config

    prepared = load_phase_config(config_path, train_file=train_file, model=model)
    train = prepared.cfg["train"]
    if bool(train.get("packing", False)):
        raise ConfigError("DPO must not pack; set train.packing: false")
    if float(train.get("beta", 0.1)) <= 0:
        raise ConfigError("DPO requires train.beta > 0")
    if str(train.get("loss_type", "sigmoid")).lower() not in {"dpo", "sigmoid"}:
        raise ConfigError(
            f"unsupported DPO loss_type {train.get('loss_type')!r}; only the "
            "standard sigmoid objective is shipped"
        )
    return train_phase(
        "dpo",
        config_path,
        time_budget=time_budget,
        max_steps=max_steps,
        run_id=run_id,
        allow_config_change=allow_config_change,
        train_file=train_file,
        model=model,
        device=device,
    )
