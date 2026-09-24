"""Unit tests for verifier._prompts module."""

from __future__ import annotations

from verifier._prompts import (
    DEFAULT_VERIFIER_SYSTEM_INSTRUCTION,
    format_fact_verification_prompt,
)


def test_default_verifier_system_instruction():
    assert "expert physician" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "binary classification" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "YES" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "NO" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "1" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "0" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION


def test_format_fact_verification_prompt():
    question = "A 45-year-old man presents with acute chest pain."
    options = {"A": "Aspirin", "B": "Morphine"}
    fact = "Aspirin inhibits platelet cyclooxygenase irreversibly."

    prompt = format_fact_verification_prompt(question, options, fact)

    assert question in prompt
    assert "A. Aspirin" in prompt
    assert "B. Morphine" in prompt
    assert fact in prompt
    assert "is_correct: YES" in prompt
    assert "NO" in prompt
    assert "rationale:" in prompt
