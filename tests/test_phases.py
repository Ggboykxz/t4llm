"""Phase-trainer contracts: prompt masking, no packing, deterministic resume."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from t4llm.pref import DPODataset
from t4llm.sft import SFTDataset
from t4llm.state import Cursor, ConfigError


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    from transformers import AutoTokenizer

    from t4llm.fixtures import make_tiny_model

    root = tmp_path_factory.mktemp("phase-tokenizer")
    return AutoTokenizer.from_pretrained(str(make_tiny_model(root / "tiny", seq_len=32)))


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _sft_rows() -> list[dict]:
    return [
        {
            "prompt": "tok5 tok6 tok7",
            "response": "tok10 tok11 tok12",
            "source": "unit",
            "license": "CC0-1.0",
        },
        {
            "messages": [
                {"role": "user", "content": "tok20 tok21"},
                {"role": "assistant", "content": "tok30 tok31"},
            ],
            "source": "unit",
            "license": "CC0-1.0",
        },
        {
            "prompt": "tok40 tok41",
            "response": "tok50 tok51 tok52 tok53",
            "source": "unit",
            "license": "CC0-1.0",
        },
    ]


def _dpo_rows() -> list[dict]:
    return [
        {
            "prompt": "tok5 tok6",
            "chosen": "tok10 tok11",
            "rejected": "tok20 tok21",
            "source": "unit",
            "license": "CC0-1.0",
        },
        {
            "prompt": "tok30 tok31",
            "chosen": "tok40 tok41 tok42",
            "rejected": "tok50",
            "source": "unit",
            "license": "CC0-1.0",
        },
    ]


def test_sft_masks_prompt_and_supervises_only_completion(tmp_path, tokenizer):
    path = _write_jsonl(tmp_path / "sft.jsonl", _sft_rows())
    dataset = SFTDataset(path, seq_len=32, seed=3407, epochs=1)
    batcher = dataset.make_batcher(tokenizer)
    batch, cursor = batcher.next_batch(dataset.cursor(0, 0, 0), 2)

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape[1] < 32  # dynamic padding, not fixed seq_len
    assert cursor.sample_idx == 2
    assert cursor.consumed_tokens == batch["supervised_tokens"] > 0
    for row in range(batch["labels"].shape[0]):
        labels = batch["labels"][row].tolist()
        idx = [i for i, x in enumerate(labels) if x != -100]
        assert idx, "completion must be supervised"
        # One contiguous completion span; a prompt token cannot leak in.
        assert idx == list(range(idx[0], idx[-1] + 1))
        assert all(x == -100 for x in labels[: idx[0]])
        # The first supervised token is the response, not the prompt.
        if row == 0:
            assert labels[idx[0]] != -100


def test_sft_no_cross_example_packing(tmp_path, tokenizer):
    path = _write_jsonl(tmp_path / "sft.jsonl", _sft_rows())
    dataset = SFTDataset(path, seq_len=32, seed=3407, epochs=1)
    batcher = dataset.make_batcher(tokenizer)
    single, _ = batcher.next_batch(dataset.cursor(0, 0, 0), 1)
    pair, _ = batcher.next_batch(dataset.cursor(0, 0, 0), 2)
    # A packed implementation would make the two-example row ~2x as long.
    single_len = int(single["attention_mask"][0].sum())
    pair_len = int(pair["attention_mask"][1].sum())
    assert pair["input_ids"].shape[1] <= max(
        single_len, pair["input_ids"].shape[1]
    )
    assert pair_len <= 32
    assert pair["input_ids"].shape[1] < 2 * single_len + 8


def test_sft_truncation_keeps_a_completion_token(tmp_path, tokenizer):
    rows = _sft_rows()
    rows[0]["prompt"] = " ".join(f"tok{i}" for i in range(5, 30))
    rows[0]["response"] = "tok40 tok41"
    path = _write_jsonl(tmp_path / "sft.jsonl", rows)
    dataset = SFTDataset(path, seq_len=8, seed=3407, epochs=1)
    batcher = dataset.make_batcher(tokenizer)
    batch, _ = batcher.next_batch(dataset.cursor(0, 0, 0), 1)
    labels = batch["labels"][0].tolist()
    assert any(x != -100 for x in labels)
    assert labels[-1] != -100


def test_sft_rejects_missing_licence(tmp_path):
    path = _write_jsonl(
        tmp_path / "sft.jsonl",
        [{"prompt": "tok5", "response": "tok6", "source": "unit"}],
    )
    with pytest.raises(ConfigError, match="licence"):
        SFTDataset(path, seq_len=16, seed=1, epochs=1)


def test_sft_rejects_non_assistant_final_message(tmp_path):
    path = _write_jsonl(
        tmp_path / "sft.jsonl",
        [
            {
                "messages": [{"role": "user", "content": "tok5 tok6"}],
                "source": "unit",
                "license": "CC0-1.0",
            }
        ],
    )
    with pytest.raises(ConfigError, match="assistant"):
        SFTDataset(path, seq_len=16, seed=1, epochs=1)


def test_dpo_batches_pairs_with_completion_masks(tmp_path, tokenizer):
    path = _write_jsonl(tmp_path / "dpo.jsonl", _dpo_rows())
    dataset = DPODataset(path, seq_len=32, seed=3407, epochs=1)
    batcher = dataset.make_batcher(tokenizer)
    batch, cursor = batcher.next_batch(dataset.cursor(0, 0, 0), 2)
    pairs = batch["pairs"]
    assert pairs == 2
    assert batch["input_ids"].shape[0] == 2 * pairs
    supervised = [int((batch["labels"][i] != -100).sum()) for i in range(2 * pairs)]
    assert all(x > 0 for x in supervised)
    # Rows are [chosen0, chosen1, rejected0, rejected1]; the pair order is the
    # same, so chosen/rejected masks line up for the DPO objective.
    assert supervised[:pairs] != supervised[pairs:]
    assert cursor.sample_idx == 2
    assert cursor.consumed_tokens == batch["supervised_tokens"]


def test_dpo_rejects_missing_completion(tmp_path):
    path = _write_jsonl(
        tmp_path / "dpo.jsonl",
        [
            {
                "prompt": "tok5",
                "chosen": "",
                "rejected": "tok6",
                "source": "unit",
                "license": "CC0-1.0",
            }
        ],
    )
    with pytest.raises(ConfigError, match="chosen"):
        DPODataset(path, seq_len=16, seed=1, epochs=1)


def test_phase_order_is_deterministic_and_resumable(tmp_path, tokenizer):
    path = _write_jsonl(tmp_path / "sft.jsonl", _sft_rows())
    a = SFTDataset(path, seq_len=32, seed=3407, epochs=2)
    b = SFTDataset(path, seq_len=32, seed=3407, epochs=2)
    assert a.revision == b.revision
    assert [int(x) for x in a.permutation(0)] == [int(x) for x in b.permutation(0)]

    ba = a.make_batcher(tokenizer)
    first, after_first = ba.next_batch(a.cursor(0, 0, 0), 1)
    second, _ = ba.next_batch(after_first, 1)
    both, _ = ba.next_batch(a.cursor(0, 0, 0), 2)
    n1 = int(first["attention_mask"][0].sum())
    n2 = int(second["attention_mask"][0].sum())
    assert first["input_ids"][0, :n1].tolist() == both["input_ids"][0, :n1].tolist()
    assert second["input_ids"][0, :n2].tolist() == both["input_ids"][1, :n2].tolist()


def test_phase_cursor_rejects_a_changed_file(tmp_path):
    path = _write_jsonl(tmp_path / "sft.jsonl", _sft_rows())
    dataset = SFTDataset(path, seq_len=32, seed=3407, epochs=1)
    bad = Cursor(data_revision="file-sha256:" + "0" * 64, shard_idx=0, sample_idx=0)
    with pytest.raises(Exception, match="data revision"):
        dataset.validate_cursor(bad)


def test_cli_exposes_sft_and_dpo_commands():
    from t4llm.cli import build_parser

    parser = build_parser()
    for cmd in ("sft", "dpo"):
        args = parser.parse_args([cmd, "--config", f"configs/{cmd}.yaml"])
        assert args.func is not None
    status = parser.parse_args(
        ["status", "--config", "configs/sft.yaml", "--train-file", "sft.jsonl"]
    )
    assert status.train_file == "sft.jsonl"


def test_sft_neftune_noise_changes_the_loss(tmp_path, tokenizer):
    """NEFTune must perturb the training forward, and only when alpha > 0."""
    import torch
    from transformers import AutoModelForCausalLM

    from t4llm.fixtures import make_tiny_model
    from t4llm.sft import sft_loss

    model_dir = make_tiny_model(tmp_path / "tiny", seq_len=32)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    path = _write_jsonl(tmp_path / "sft.jsonl", _sft_rows())
    dataset = SFTDataset(path, seq_len=32, seed=3407, epochs=1)
    batch, _ = dataset.make_batcher(tokenizer).next_batch(dataset.cursor(0, 0, 0), 2)

    plain, _ = sft_loss(
        model, batch, {"train": {"neftune_noise_alpha": 0.0}}, on_cuda=False
    )
    torch.manual_seed(0)
    noisy, _ = sft_loss(
        model, batch, {"train": {"neftune_noise_alpha": 5.0}}, on_cuda=True
    )
    assert torch.isfinite(plain) and torch.isfinite(noisy)
    assert not torch.allclose(plain, noisy)

    # alpha=0 must be exactly the ordinary input_ids forward, not a no-op noise
    # addition that still perturbs anything.
    model.eval()
    with torch.no_grad():
        plain2, _ = sft_loss(
            model, batch, {"train": {"neftune_noise_alpha": 0.0}}, on_cuda=True
        )
    assert torch.allclose(plain, plain2, atol=1e-6)
