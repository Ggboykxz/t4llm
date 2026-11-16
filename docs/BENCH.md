# Benchmarks

All numbers below are measured by `python -m t4llm bench` on the actual runtime. Unmeasured rows are marked SKIPPED, never guessed.

| UTC | GPU | torch | model | mode | seq | micro-bs | tok/s | VRAM peak GiB | ckpt bytes | upload s | upload Mbps | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-11-16T13:11:24+00:00 | Tesla T4 | 2.14.1+cu130 | Qwen/Qwen3-1.7B-Base | LoRA fp16 r32 | 2048 | 2 | 1027.2 | 12.02 | 222252479 | SKIPPED | SKIPPED | unsloth unavailable; using standard non-reentrant HF checkpointing; upload not measured (no --upload-repo/HF_TOKEN) |

## GGUF conversion (measured, base model only)

This verifies that `llama.cpp`'s converter accepts the target architecture and
tokenizer. It is **not** a fine-tuned export and not a quantization benchmark.

| UTC | model | outtype | GGUF bytes | seconds | magic |
|---|---|---|---:|---:|---|
| 2026-11-16T13:54:54+00:00 | Qwen/Qwen3-1.7B-Base | f16 | 3447348960 | 46.3 | `GGUF` |

The tiny test fixture uses a synthetic `WordLevel` tokenizer, which llama.cpp
does not recognise as a BPE pre-tokenizer. GGUF conversion is therefore tested
on a real model directory; the fixture covers the merge/safetensors path with
`--no-gguf`.

## Merged SFT export (measured, local transport)

One SFT step on `Qwen/Qwen3-0.6B-Base` (4 licence-declaring records, seq 128,
LoRA r=8/alpha=16), then a full merge to fp16 safetensors and f16 GGUF. The
export was published to the local test backend only, not to a real Hub repo.

| UTC | base | phase/step | train s | merged safetensors bytes | GGUF f16 bytes | total s | safetensors dtype | adapter keys |
|---|---|---|---:|---:|---:|---:|---|---:|
| 2026-11-16T13:57:50+00:00 | Qwen/Qwen3-0.6B-Base | sft/1 | 23.7 | 1192134784 | 1198182080 | 102.6 | fp16 | 0 |

The GGUF magic is `GGUF`, the export manifest lists 8 files, and `SHA256SUMS`
covers every file except itself. This is a smoke test, not a quality result:
one step is not a trained model.
