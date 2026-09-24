"""Parser module for extracting binary fact verification judgments (0 or 1) and rationales."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_VERDICT_MAP: dict[str, int] = {
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0,
    "yes": 1,
    "no": 0,
    "correct": 1,
    "incorrect": 0,
    "accurate": 1,
    "inaccurate": 0,
    "wrong": 0,
}

_VERDICT_KEYS = (
    "is_correct",
    "verdict",
    "truthfulness",
    "label",
    "correct",
    "binary_classification",
    "classification",
    "result",
    "score",
)
_RATIONALE_KEYS = ("rationale", "explanation", "reason", "reasoning")

_KV_PATTERN = re.compile(
    rf"(?:{'|'.join(_VERDICT_KEYS)})\s*[:=]\s*({'|'.join(_VERDICT_MAP)})\b",
    re.IGNORECASE,
)
_RATIONALE_PATTERN = re.compile(
    rf"(?:{'|'.join(_RATIONALE_KEYS)})\s*[:=]\s*(.+)",
    re.IGNORECASE,
)
_STANDALONE_PATTERN = re.compile(
    r"\b([01]|correct|incorrect|true|false|yes|no)\b",
    re.IGNORECASE,
)


def _coerce_verdict(val: Any) -> int | None:
    """Coerce boolean, integer, or verdict string/enum to 0 or 1."""
    if val is None:
        return None
    return _VERDICT_MAP.get(str(val).strip().lower())


def _extract_json(text: str) -> tuple[int, str] | None:
    """Extract verdict and rationale from JSON object enclosed in braces."""
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        try:
            data = json.loads(text[first : last + 1])
            if isinstance(data, dict):
                verdict = next(
                    (
                        v
                        for k in _VERDICT_KEYS
                        if (v := _coerce_verdict(data.get(k))) is not None
                    ),
                    None,
                )
                if verdict is not None:
                    rationale = next(
                        (str(data[k]).strip() for k in _RATIONALE_KEYS if data.get(k)),
                        "",
                    )
                    return verdict, rationale
        except json.JSONDecodeError:
            pass
    return None


def _extract_key_value(text: str) -> tuple[int, str] | None:
    """Extract verdict and rationale from key-value text patterns."""
    if match := _KV_PATTERN.search(text):
        verdict = _VERDICT_MAP[match.group(1).lower()]
        rat_match = _RATIONALE_PATTERN.search(text)
        rationale = rat_match.group(1).strip() if rat_match else text
        return verdict, rationale
    return None


def _extract_standalone(text: str) -> tuple[int, str] | None:
    """Extract standalone verdict keyword from text."""
    if match := _STANDALONE_PATTERN.search(text):
        verdict = _VERDICT_MAP[match.group(1).lower()]
        rationale = "" if text.strip().lower() in ("0", "1", "yes", "no") else text
        return verdict, rationale
    return None


def parse_fact_verification(raw_response: str) -> tuple[int, str]:
    """Parse raw LLM-as-a-judge response into binary verdict (0 or 1) and rationale.

    Args:
        raw_response: Raw text returned by the verifier model.

    Returns:
        tuple of (verdict: int [0 or 1], rationale: str).
    """
    if not raw_response or not raw_response.strip():
        logger.warning("Empty response received for fact verification.")
        return 0, "Empty response from verifier."

    cleaned = raw_response.strip()
    search_text = (
        re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip() or cleaned
    )

    result = (
        _extract_json(search_text)
        or _extract_key_value(search_text)
        or _extract_standalone(search_text)
    )
    if result is not None:
        return result

    logger.warning(
        "Could not extract binary verdict from verifier response: %r. Defaulting to 0.",
        cleaned[:100],
    )
    return 0, f"Unparseable response: {cleaned[:200]}"


__all__ = [
    "parse_fact_verification",
]
