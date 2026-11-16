# Benchmarks

All numbers below are measured by `python -m t4llm bench` on the actual runtime. Unmeasured rows are marked SKIPPED, never guessed.

| UTC | GPU | torch | model | mode | seq | micro-bs | tok/s | VRAM peak GiB | ckpt bytes | upload s | upload Mbps | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-11-16T13:11:24+00:00 | Tesla T4 | 2.14.1+cu130 | Qwen/Qwen3-1.7B-Base | LoRA fp16 r32 | 2048 | 2 | 1027.2 | 12.02 | 222252479 | SKIPPED | SKIPPED | unsloth unavailable; using standard non-reentrant HF checkpointing; upload not measured (no --upload-repo/HF_TOKEN) |
