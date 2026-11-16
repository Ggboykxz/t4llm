"""t4llm — crash-only domain specialization on a single free T4.

Importing this package sets the process environment required by the contract
before CUDA is initialised.  It deliberately does not import torch.
"""
from __future__ import annotations

import os

__version__ = "0.1.0"

# §4 environment contract.  setdefault so an operator can override explicitly.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "offline")
# Required by cuBLAS for deterministic reductions; must be set before CUDA init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
