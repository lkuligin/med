"""Parser and extraction module for medical facts and multiple-choice answers."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from inference._schemas import MedicalFacts
from one_shot.parser import evaluate_prediction, extract_predicted_option

logger = logging.getLogger(__name__)


def _extract_facts_from_obj(obj: Any) -> list[str]:
    """Extract list of non-empty fact strings from a dictionary or list."""
    if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
        return [str(f).strip() for f in obj["facts"] if str(f).strip()]
    if isinstance(obj, list):
        return [str(f).strip() for f in obj if str(f).strip()]
    return []


def _extract_json_facts(text: str) -> list[str]:
    """Attempt extraction of facts from structured JSON in text."""
    try:
        parsed = MedicalFacts.model_validate_json(text)
        if parsed.facts:
            return [f.strip() for f in parsed.facts if f and f.strip()]
    except Exception:
        pass

    patterns = [
        r"```(?:json)?\s*([{\[][\s\S]*?[}\]])\s*```",
        r"(\{[\s\S]*\})",
        r"(\[[\s\S]*\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                data = json.loads(match.group(1))
                facts = _extract_facts_from_obj(data)
                if facts:
                    return facts
            except Exception:
                continue

    return []


def _extract_bullet_facts(text: str) -> list[str]:
    """Fallback extraction of facts from bulleted or numbered lines."""
    facts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("{", "}", "[", "]")):
            continue
        cleaned = re.sub(r"^(?:[\*\-\•]|\d+[\.\)])\s*", "", line).strip()
        if len(cleaned) > 5:
            facts.append(cleaned)
    return facts


def parse_medical_facts(response_text: str) -> list[str]:
    """Parse atomic medical facts from LLM response text.

    Supports:
    1. Direct JSON conforming to MedicalFacts schema
    2. Markdown code fences ```json ... ```
    3. Embedded JSON objects or arrays
    4. Bulleted or numbered fallback lines
    """
    if not response_text or not response_text.strip():
        return []

    cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    search_text = cleaned or response_text.strip()

    return _extract_json_facts(search_text) or _extract_bullet_facts(search_text)


__all__ = [
    "evaluate_prediction",
    "extract_predicted_option",
    "parse_medical_facts",
]
