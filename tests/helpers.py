from __future__ import annotations

from pathlib import Path

from t4llm.state import Cursor, build_manifest, utcnow


def make_fake_checkpoint(
    root: str | Path,
    *,
    step: int,
    run_id: str = "tiny-cpt-a1b2c3d4",
    config_hash: str = "a1b2c3d4",
    tier: str = "full",
    loss_ema: float = 4.0,
    seq_len: int = 64,
    shard_idx: int = 0,
    sample_idx: int | None = None,
) -> Path:
    if sample_idx is None:
        sample_idx = step
    d = Path(root) / f"ckpt-{step:08d}"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"fake-safetensors-" + str(step).encode())
    (d / "adapter_config.json").write_text('{"r": 4}')
    (d / "config.json").write_text('{"run": {"phase": "cpt"}}')
    (d / "env.json").write_text('{"torch_version": "test"}')
    (d / "trainer_state.json").write_text(
        f'{{"global_step": {step}, "consumed_tokens": {step * seq_len}}}'
    )
    (d / "tokenizer").mkdir()
    (d / "tokenizer" / "tokenizer.json").write_text("{}")
    if tier == "full":
        (d / "optimizer.pt").write_bytes(b"fake-optimizer-" + str(step).encode())
        (d / "scheduler.pt").write_bytes(b"fake-scheduler")
        (d / "rng_state.pt").write_bytes(b"fake-rng")
    cursor = Cursor(
        data_revision="d" * 64,
        shard_idx=shard_idx,
        sample_idx=sample_idx,
        consumed_tokens=sample_idx * seq_len,
    )
    build_manifest(
        d,
        run_id=run_id,
        global_step=step,
        tier=tier,
        consumed_tokens=cursor.consumed_tokens,
        cursor=cursor,
        config_hash=config_hash,
        loss_ema=loss_ema,
        created_utc=utcnow(),
    )
    return d
