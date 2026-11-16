#!/usr/bin/env bash
# Idempotent environment setup.  Target: < 120 s when the Colab image already
# has torch.  Never reinstalls or upgrades torch/torchvision/torchaudio.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "[bootstrap] python: $($PYTHON -c 'import sys; print(sys.version.split()[0])')"
$PYTHON - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("t4llm requires Python >= 3.10")
try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch is missing from the base image: {exc}") from exc
print(f"[bootstrap] torch: {torch.__version__} cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[bootstrap] WARNING: CUDA is not available; training will refuse to start")
PY

# Only install the pinned direct dependencies.  --upgrade-strategy only-if-needed
# keeps an already-correct Colab image from being churned.
echo "[bootstrap] installing pinned dependencies"
$PYTHON -m pip install --quiet --disable-pip-version-check \
  --upgrade-strategy only-if-needed -r requirements.lock

# llama.cpp's converter is pure Python; no CUDA build is needed to emit f16 GGUF.
if [[ ! -f "${LLAMA_CPP_DIR:-/content/llama.cpp}/convert_hf_to_gguf.py" ]]; then
  if [[ "${T4LLM_SKIP_LLAMA_CPP:-0}" != "1" ]]; then
    echo "[bootstrap] fetching llama.cpp converter (shallow)"
    if command -v git >/dev/null 2>&1; then
      git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR:-/content/llama.cpp}" \
        >/dev/null 2>&1 || true
      if [[ -d "${LLAMA_CPP_DIR:-/content/llama.cpp}/.git" ]]; then
        git -C "${LLAMA_CPP_DIR:-/content/llama.cpp}" sparse-checkout set \
          --skip-checks convert_hf_to_gguf.py conversion gguf-py >/dev/null 2>&1 || true
      fi
    fi
  fi
fi
if [[ -f "${LLAMA_CPP_DIR:-/content/llama.cpp}/convert_hf_to_gguf.py" ]]; then
  $PYTHON -m pip install --quiet --disable-pip-version-check gguf \
    || echo "[bootstrap] WARNING: gguf install failed; export --no-gguf still works"
else
  echo "[bootstrap] WARNING: llama.cpp converter unavailable; export --no-gguf works"
fi

echo "[bootstrap] capability report"
$PYTHON -m t4llm env || {
  echo "[bootstrap] environment is not training-capable; see report above"
  exit 1
}

echo "[bootstrap] contract self-test (CPU-safe subset)"
$PYTHON -m pytest tests -m "not gpu and not slow" -q

echo "[bootstrap] done"
