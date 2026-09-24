"""Unit tests for verifier.parser module."""

from __future__ import annotations

from verifier.parser import parse_fact_verification


def test_parse_json_valid_one():
    raw = '{"is_correct": 1, "rationale": "Directly supported by Harrison\'s Principles."}'
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 1
    assert "Harrison's" in rationale


def test_parse_json_valid_zero():
    raw = '{"is_correct": 0, "rationale": "Cisplatin is not an aminoglycoside."}'
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 0
    assert "Cisplatin" in rationale


def test_parse_json_in_markdown_code_fence():
    raw = (
        "Here is the evaluation:\n"
        "```json\n"
        "{\n"
        '  "is_correct": 1,\n'
        '  "rationale": "Accurate clinical anatomy."\n'
        "}\n"
        "```"
    )
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 1
    assert "Accurate clinical anatomy." in rationale


def test_parse_json_alternate_keys():
    # verdict key with boolean True
    raw = '{"verdict": true, "explanation": "Factually true."}'
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 1
    assert "Factually true." in rationale

    # label key with 0
    raw2 = '{"label": 0, "reason": "Incorrect mechanism."}'
    verdict2, rationale2 = parse_fact_verification(raw2)
    assert verdict2 == 0
    assert "Incorrect mechanism." in rationale2


def test_parse_regex_key_value():
    raw = (
        "Verdict: 1\nRationale: This pharmacological statement is completely accurate."
    )
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 1
    assert "pharmacological" in rationale

    raw2 = "is_correct: 0\nExplanation: Contradicts medical guidelines."
    verdict2, rationale2 = parse_fact_verification(raw2)
    assert verdict2 == 0
    assert "Contradicts" in rationale2


def test_parse_single_digits():
    v1, _ = parse_fact_verification("1")
    assert v1 == 1

    v0, _ = parse_fact_verification("0")
    assert v0 == 0


def test_parse_empty_and_garbage():
    v_empty, rat_empty = parse_fact_verification("")
    assert v_empty == 0
    assert "Empty" in rat_empty

    v_garb, rat_garb = parse_fact_verification("Unrelated conversational response.")
    assert v_garb == 0


def test_parse_standalone_words():
    v_true, _ = parse_fact_verification("The statement is correct.")
    assert v_true == 1

    v_false, _ = parse_fact_verification("The statement is incorrect.")
    assert v_false == 0


def test_parse_json_no_rationale():
    raw = '{"is_correct": 1}'
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 1
    assert rationale == ""


def test_parse_malformed_json_fallback():
    raw = (
        '{"is_correct": 1, invalid json text}\nverdict: 1\nrationale: Fallback parsed.'
    )
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 1
    assert "Fallback parsed." in rationale


def test_parse_with_think_block():
    raw = (
        "<think>\n"
        "The fact might seem incorrect at first glance, but let me check.\n"
        "Actually, it is true.\n"
        "</think>\n"
        '{"is_correct": 1, "rationale": "Accurate mechanism."}'
    )
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 1
    assert "Accurate mechanism." in rationale


def test_parse_json_enum_yes_and_no():
    raw_yes = '{"is_correct": "YES", "rationale": "Strongly supported by guidelines."}'
    verdict, rationale = parse_fact_verification(raw_yes)
    assert verdict == 1
    assert "guidelines" in rationale

    raw_no = '{"is_correct": "NO", "rationale": "Contraindicated in renal failure."}'
    verdict, rationale = parse_fact_verification(raw_no)
    assert verdict == 0
    assert "Contraindicated" in rationale


def test_parse_truthfulness_key():
    raw = '{"truthfulness": "YES", "rationale": "Correct pathophysiology."}'
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 1
    assert "pathophysiology" in rationale


def test_parse_standalone_yes_no():
    v_yes, rat_yes = parse_fact_verification("YES")
    assert v_yes == 1
    assert rat_yes == ""

    v_no, rat_no = parse_fact_verification("NO")
    assert v_no == 0
    assert rat_no == ""

    v_yes_lower, _ = parse_fact_verification("yes")
    assert v_yes_lower == 1

    v_no_lower, _ = parse_fact_verification("no")
    assert v_no_lower == 0


def test_parse_kv_yes_no():
    raw = "is_correct: YES\nrationale: Correct statement."
    verdict, rationale = parse_fact_verification(raw)
    assert verdict == 1
    assert "Correct statement." in rationale

    raw2 = "truthfulness: NO\nrationale: Misleading claim."
    verdict2, rationale2 = parse_fact_verification(raw2)
    assert verdict2 == 0
    assert "Misleading claim." in rationale2
