"""Unit tests for atomic-fact extraction.

Free-form fact generation - what `--free-form-facts` turns on for models that
answer a strict schema with an empty list - lets the model choose how it wraps
each fact. These cover the wrappings seen on the cards.
"""

from __future__ import annotations

from inference.parser import parse_medical_facts


def test_plain_string_facts():
    text = '{"facts": ["Aspirin inhibits COX-1.", "COX-1 makes thromboxane."]}'
    assert parse_medical_facts(text) == ["Aspirin inhibits COX-1.",
                                         "COX-1 makes thromboxane."]


def test_facts_wrapped_in_objects_yield_their_sentence():
    """The shape gpt-oss-20b answers with when it is given no schema."""
    text = """```json
    {"facts": [{"id": 1, "statement": "Aspirin inhibits COX-1.",
                "type": "pharmacology"},
               {"id": 2, "statement": "COX-1 makes thromboxane.",
                "type": "physiology"}]}
    ```"""
    assert parse_medical_facts(text) == ["Aspirin inhibits COX-1.",
                                         "COX-1 makes thromboxane."]


def test_alternative_statement_keys():
    text = ('{"facts": [{"fact": "One."}, {"text": "Two."}, '
            '{"claim": "Three."}, {"content": "Four."}]}')
    assert parse_medical_facts(text) == ["One.", "Two.", "Three.", "Four."]


def test_object_without_a_sentence_key_is_kept_rather_than_dropped():
    """Losing a fact silently would let a candidate pass on fewer checks."""
    facts = parse_medical_facts('{"facts": [{"subject": "patient",'
                                ' "attribute": "age", "value": "27 years"}]}')
    assert len(facts) == 1
    assert "27 years" in facts[0]


def test_bare_list_of_objects():
    assert parse_medical_facts('[{"statement": "One."}, "Two."]') == ["One.",
                                                                      "Two."]


def test_empty_and_blank_facts_are_dropped():
    assert parse_medical_facts('{"facts": ["", "  ", {"statement": " "}]}') == []


def test_thinking_block_is_stripped_before_parsing():
    text = ('<think>the model deliberates about {"facts": ["wrong"]}</think>'
            '{"facts": [{"statement": "Right."}]}')
    assert parse_medical_facts(text) == ["Right."]


def test_bullet_fallback_when_there_is_no_json():
    text = "- Aspirin inhibits COX-1.\n2) COX-1 makes thromboxane.\n"
    assert parse_medical_facts(text) == ["Aspirin inhibits COX-1.",
                                         "COX-1 makes thromboxane."]


def test_no_facts_at_all():
    assert parse_medical_facts("") == []
    assert parse_medical_facts('{"facts": []}') == []


def test_fact_string_that_is_itself_json_is_unwrapped():
    """The schema asks for strings; it does not ask them to be sentences."""
    text = ('{"facts": ["{\\"category\\": \\"demographics\\", '
            '\\"statement\\": \\"The patient is 68 years old.\\"}"]}')
    assert parse_medical_facts(text) == ["The patient is 68 years old."]


def test_a_sentence_that_merely_starts_with_a_brace_survives():
    text = '{"facts": ["{not json} is how the note was written."]}'
    assert parse_medical_facts(text) == ["{not json} is how the note was written."]


def test_facts_survive_a_list_left_without_its_closing_bracket():
    # gpt-oss-20b, free-form: '..."}' where '..."]}' belongs.
    text = '{"facts":["Migraine is often unilateral.", "SAH presents with a thunderclap headache."}'
    assert parse_medical_facts(text) == ["Migraine is often unilateral.",
                                         "SAH presents with a thunderclap headache."]


def test_facts_survive_a_last_fact_closed_with_a_typographic_quote():
    text = '{"facts":["Migraine is often unilateral.","Papilledema signals raised pressure.\u201d}'
    assert parse_medical_facts(text) == ["Migraine is often unilateral.",
                                         "Papilledema signals raised pressure."]


def test_a_list_cut_off_mid_fact_keeps_the_facts_before_the_cut():
    assert parse_medical_facts('{"facts":["SAH is sudden.","Migraine is often unilat') == [
        "SAH is sudden."]
    assert parse_medical_facts('{"facts":["Migraine is often unilat') == []


def test_facts_survive_a_stray_escape_before_the_separator():
    # gpt-oss-20b: 'life.\\",' - the quote that should close the fact is escaped.
    text = '{"facts":["SAH is the \\"worst headache of life.\\","Migraine is unilateral."]}'
    assert parse_medical_facts(text) == ['SAH is the "worst headache of life."',
                                         "Migraine is unilateral."]
