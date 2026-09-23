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


def register_gateway_names() -> None:
    """Teach llm_monkeys that our gateway names are already canonical.

    resolve_model_name() turns anything it does not recognise into a Vertex AI
    identifier, so "gpt-oss-120b" would become
    "vertex_ai/openai/gpt-oss-120b-maas" - an address the gateway does not
    serve, and the wrong model in every summary.

    Done here rather than where models are built, because the name is read
    long before that: a script prints which model it is about to call, and
    that line has to say the same thing the call does. Idempotent, and it
    never overrides a name the reference already knows.
    """
    from config import SUPPORTED_MODELS

    from gdanschin_runtime.models import BASE_MODELS, JUDGE_MODELS

    for entry in list(BASE_MODELS.values()) + list(JUDGE_MODELS.values()):
        SUPPORTED_MODELS.setdefault(entry.gateway_model.lower(), entry.gateway_model)


ensure_on_path()
register_gateway_names()

__all__ = ["LLM_MONKEYS_ROOT", "ensure_on_path", "register_gateway_names"]
