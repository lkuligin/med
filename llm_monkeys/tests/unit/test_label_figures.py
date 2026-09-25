"""Screening questions for exhibits the text does not include."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dataset import MedQAQuestion
import label_figures
from label_figures import (
    LABELS,
    FigureLabelConfig,
    FigureLabelWorkflow,
    default_output,
    format_prompt,
    parse_label,
    read_labels,
    write_labels,
)


def question(qid: str = "1", text: str = "What does the ECG show?") -> MedQAQuestion:
    return MedQAQuestion(
        question_id=qid,
        question=text,
        options={"B": "Second", "A": "First"},
        answer_idx="A",
        answer="First",
    )


def test_the_judge_is_sampled_the_way_gemini_is_meant_to_be():
    # Not the generator's 0.8: this is a judgement about a question, made by
    # the model the pipeline judges facts with.
    config = FigureLabelConfig()
    assert config.temperature == 1.0
    assert config.model_name == "gemini-3.8-flash"
    assert config.max_tokens == 16384


def test_the_prompt_carries_the_options_in_order():
    prompt = format_prompt(question())
    assert "What does the ECG show?" in prompt
    # "an exhibit is mentioned but the text still decides" can only be told
    # from "the exhibit decides" by reading what there is to choose between.
    assert prompt.index("A. First") < prompt.index("B. Second")
    for label in LABELS:
        assert label in prompt


@pytest.mark.parametrize("raw, expected", [
    ('{"label": "no_figure", "reason": "All in the text."}', ("no_figure", "All in the text.")),
    ('```json\n{"label": "requires_figure", "reason": "ECG."}\n```',
     ("requires_figure", "ECG.")),
    ('Sure! {"label": "mentions_figure_answerable", "reason": "x"} hope that helps',
     ("mentions_figure_answerable", "x")),
    ('{"label": "maybe", "reason": "x"}', ("", "x")),      # not one of the three
    ('{"label": "no_figure"}', ("no_figure", "")),
    ("no json at all", ("", "")),
    ('{"label": "no_figure", ', ("", "")),                  # truncated
    ("", ("", "")),
])
def test_a_reply_is_read_or_refused(raw, expected):
    assert parse_label(raw) == expected


def test_the_file_is_named_as_the_difficult_lists_are():
    assert default_output("bigbio/med_qa").name == "figure_labels.csv"
    assert default_output("mkieffer/Medbullets").name == "figure_labels_mb.csv"


def test_labels_survive_a_round_trip_in_dataset_order(tmp_path):
    path = tmp_path / "labels.csv"
    questions = [question("003"), question("001"), question("002")]
    labels = {"001": ("no_figure", "a"), "002": ("requires_figure", "b"),
              "003": ("mentions_figure_answerable", "c")}

    write_labels(path, questions, labels)

    assert [row.split(",")[0] for row in path.read_text().splitlines()[1:]] == \
        ["003", "001", "002"], "the file follows the dataset, not the sort order"
    assert read_labels(path) == labels


def test_a_padded_id_stays_padded(tmp_path):
    path = tmp_path / "labels.csv"
    write_labels(path, [question("007")], {"007": ("no_figure", "")})
    assert "007" in read_labels(path)
    assert "7" not in read_labels(path)


def test_an_errored_row_is_labelled_again_next_time(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("question_id,label,reason\n1,error,timed out\n2,no_figure,fine\n")
    # Keeping the error would freeze it into the file for good.
    assert read_labels(path) == {"2": ("no_figure", "fine")}


@pytest.mark.asyncio
async def test_a_reply_without_a_label_is_retried_then_given_up_on():
    workflow = FigureLabelWorkflow(config=FigureLabelConfig(max_retries=2),
                                   runner=MagicMock())
    workflow._ask = AsyncMock(return_value="I cannot tell")

    label, reason = await workflow.label(question())

    assert label == ""
    assert workflow._ask.await_count == 3, "the first try plus max_retries"


@pytest.mark.asyncio
async def test_every_question_is_labelled_once():
    workflow = FigureLabelWorkflow(config=FigureLabelConfig(concurrency=2),
                                   runner=MagicMock())
    workflow._ask = AsyncMock(
        return_value='{"label": "no_figure", "reason": "text only"}')

    labels = await workflow.run([question("1"), question("2"), question("3")])

    assert labels == {q: ("no_figure", "text only") for q in ("1", "2", "3")}
    assert workflow._ask.await_count == 3
