from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import InferenceConfig
from dataset import MedQAQuestion
from one_shot.workflow import OneShotInferenceWorkflow


@pytest.fixture
def sample_questions() -> list[MedQAQuestion]:
    """Returns two standard sample questions for multi-question workflow tests."""
    return [
        MedQAQuestion(
            question_id="0",
            question="Sample question 1",
            options={"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
            answer_idx="A",
            answer="Opt A",
            meta_info="step1",
        ),
        MedQAQuestion(
            question_id="1",
            question="Sample question 2",
            options={"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
            answer_idx="B",
            answer="Opt B",
            meta_info="step2",
        ),
    ]


@pytest.fixture
def single_question() -> MedQAQuestion:
    """Returns a single question for targeted retry/parse tests."""
    return MedQAQuestion(
        question_id="test_retry_q",
        question="Which drug causes ototoxicity?",
        options={
            "A": "Cisplatin",
            "B": "Paracetamol",
            "C": "Amoxicillin",
            "D": "Metformin",
        },
        answer_idx="A",
        answer="Cisplatin",
    )


@pytest.fixture
def workflow_factory():
    """Factory fixture to create OneShotInferenceWorkflow with common mocks pre-configured."""

    def _create(
        config: InferenceConfig | None = None,
        runner: Any = None,
        agent: Any = None,
        session_service: Any = None,
        **config_kwargs: Any,
    ) -> OneShotInferenceWorkflow:
        if config is None:
            config_kwargs.setdefault("output_filepath", "")
            config = InferenceConfig(**config_kwargs)
        if session_service is None:
            session_service = MagicMock()
            session_service.create_session = AsyncMock()
        return OneShotInferenceWorkflow(
            config=config,
            agent=agent or MagicMock(),
            runner=runner or MagicMock(),
            session_service=session_service,
        )

    return _create
