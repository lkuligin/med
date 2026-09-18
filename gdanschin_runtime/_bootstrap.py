"""Make llm_monkeys modules importable from gdanschin_runtime.

llm_monkeys is not a package: its modules import each other as top-level names
(`from config import ...`, `from dataset import ...`), so the llm_monkeys/
directory itself must sit on sys.path as a root. The project already
acknowledges this - see llm_monkeys/verifier/cli.py, which patches the path by
hand.

This is the single place where we do it. Import it first in any
gdanschin_runtime module that needs llm_monkeys:

    from gdanschin_runtime import _bootstrap  # noqa: F401
    from config import CandidateInferenceConfig

Caution: module names inside llm_monkeys are very generic (config, cli,
dataset). The path is appended to the END of sys.path so it cannot shadow
third-party packages of the same name. If llm_monkeys ends up shadowed
instead, switch to insert(0) and verify nothing else broke.
"""

from __future__ import annotations

import sys
from pathlib import Path

LLM_MONKEYS_ROOT = Path(__file__).resolve().parents[1] / "llm_monkeys"


def ensure_on_path() -> Path:
    """Add llm_monkeys to sys.path (idempotent) and return its path."""
    if not LLM_MONKEYS_ROOT.is_dir():
        raise RuntimeError(f"llm_monkeys not found at: {LLM_MONKEYS_ROOT}")
    root = str(LLM_MONKEYS_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    return LLM_MONKEYS_ROOT


ensure_on_path()

__all__ = ["LLM_MONKEYS_ROOT", "ensure_on_path"]
