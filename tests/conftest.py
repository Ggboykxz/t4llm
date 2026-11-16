from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("T4LLM_ALLOW_NON_T4", "1")
os.environ.setdefault("T4LLM_ALLOW_SMALL_BATCH", "1")
os.environ.setdefault("T4LLM_ALLOW_SYNTHETIC", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
