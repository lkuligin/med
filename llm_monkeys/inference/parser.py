"""Parser and extraction module for medical facts and multiple-choice answers."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from inference._schemas import MedicalFacts
from one_shot.parser import evaluate_prediction, extract_predicted_option

_FACTS_USED = re.compile(r"FACTS\s+USED\s*:\s*\[?([0-9,\s]*)\]?", re.IGNORECASE)


def extract_cited_facts(response_text: str, n_facts: int) -> list[int] | None:
    """Indices (0-based) of the facts an answer says it relies on.

    Reads the last "FACTS USED: 1, 4, 7" line, whose numbers are the 1-based
    ones the answer step was shown. Numbers outside 1..n_facts are dropped.
    None when the answer has no such line, so "cited nothing" and "did not
    say" stay apart.
    """
    matches = _FACTS_USED.findall(response_text or "")
    if not matches:
        return None
    numbers = [int(n) for n in re.findall(r"\d+", matches[-1])]
    return sorted({n - 1 for n in numbers if 1 <= n <= n_facts})

logger = logging.getLogger(__name__)


# What a model calls the sentence when it wraps each fact in an object of its
# own. Everything else it puts there - an id, a category, a confidence - is
# bookkeeping about the fact rather than the fact itself.
_STATEMENT_KEYS = ("statement", "fact", "text", "claim", "content")


def _unwrap_json_string(text: str) -> str:
    """A fact the model serialised before putting it in the list.

    The schema asks for strings and gets them, so nothing rejects
    ``"{\\"statement\\": \\"...\\", \\"type\\": \\"demographic\\"}"`` - and the judge
    is then asked to verify a JSON object. 3.5% of gpt-oss-20b's candidates
    carry at least one.
    """
    if not text.startswith("{"):
        return text
    try:
        inner = json.loads(text)
    except Exception:
        return text
    return _fact_text(inner) if isinstance(inner, dict) else text


def _fact_text(item: Any) -> str:
    """The sentence in one list entry, however the model wrapped it.

    Asked without a schema, models answer
    ``{"facts": [{"id": 1, "statement": "...", "type": "..."}]}`` as readily as
    a list of strings. str() on that entry yields a Python repr, which then
    travels to the judge as the fact to verify.
    """
    if isinstance(item, dict):
        for key in _STATEMENT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # An unfamiliar wrapping, so keep the whole entry rather than drop it:
        # a lost fact is one check fewer, and a candidate that passes on fewer
        # checks than it should. Only a wrapping with nothing in it goes.
        if not any(isinstance(v, str) and v.strip() for v in item.values()):
            return ""
    return _unwrap_json_string(str(item).strip())


def _extract_facts_from_obj(obj: Any) -> list[str]:
    """Extract list of non-empty fact strings from a dictionary or list."""
    if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
        return [f for f in map(_fact_text, obj["facts"]) if f]
    if isinstance(obj, list):
        return [f for f in map(_fact_text, obj) if f]
    return []


def _extract_json_facts(text: str) -> list[str]:
    """Attempt extraction of facts from structured JSON in text."""
    try:
        parsed = MedicalFacts.model_validate_json(text)
        if parsed.facts:
            # Through _fact_text as well: the schema constrains the list to
            # strings, not the strings to being sentences.
            facts = [f for f in map(_fact_text, parsed.facts) if f]
            if facts:
                return facts
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
    "extract_cited_facts",
    "extract_predicted_option",
    "parse_medical_facts",
]
