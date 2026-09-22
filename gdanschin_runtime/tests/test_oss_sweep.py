"""That the sweep judges exactly the models it is meant to.

Stage 3 runs a paid frontier model over every candidate of every question.
An accidental run produces verdicts indistinguishable from intended ones and
a bill nobody planned, so the restriction is worth a test rather than care.
"""

import pytest

from gdanschin_runtime import oss_sweep
from gdanschin_runtime.models import BASE_MODELS

# Written out rather than derived from SWEEP: a test that reads the same
# table it checks would pass no matter how that table changed.
MEANT_TO_BE_JUDGED = {
    "qwen3.6-35b-a3b-nr-local",
    "qwen3.5-9b-nr-local",
    "qwen3.5-4b-nr-local",
}

EXPECTED_K = {
    "qwen3.6-35b-a3b-nr-local": 50,
    "qwen3.5-9b-nr-local": 100,
    "qwen3.5-4b-nr-local": 100,
    "gemma-4-e2b-local": 100,
    "gpt-oss-20b-local": 100,
    "qwen3.6-27b-nr-local": 50,
    "qwen3.8-27b-nr-local": 50,
    "qwen3.5-2b-nr-local": 100,
    "qwen3.5-0.8b-nr-local": 100,
}


def test_exactly_the_named_models_are_judged():
    assert {t.base for t in oss_sweep.SWEEP if t.judged} == MEANT_TO_BE_JUDGED


def test_every_model_keeps_the_candidate_count_it_was_given():
    assert {t.base: t.k for t in oss_sweep.SWEEP} == EXPECTED_K


@pytest.mark.parametrize("target", [t for t in oss_sweep.SWEEP if not t.judged],
                         ids=lambda t: t.base)
def test_an_unjudged_model_cannot_be_judged_by_accident(target):
    with pytest.raises(oss_sweep.NotJudged, match="costs money"):
        oss_sweep.verdicts_command(target, oss_sweep.MEDQA, "python")


@pytest.mark.parametrize("target", [t for t in oss_sweep.SWEEP if t.judged],
                         ids=lambda t: t.base)
def test_a_judged_model_builds_a_verdicts_command(target):
    command = oss_sweep.verdicts_command(target, oss_sweep.MEDQA, "python")
    assert "verifier.cli" in command
    assert command[command.index("--run-name") + 1] == target.base


def test_every_model_in_the_sweep_is_served_locally():
    """A gateway entry here would spend quota on generation as well, and the
    point of the sweep is that generation runs on our own cards."""
    for target in oss_sweep.SWEEP:
        entry = BASE_MODELS[target.base]
        assert entry.base_url, f"{target.base} is not a locally served entry"


def test_candidates_are_asked_of_the_local_endpoint(  ):
    target = oss_sweep.SWEEP[0]
    command = oss_sweep.candidates_command(target, oss_sweep.MEDBULLETS, "python")
    assert "inference.cli" in command
    assert command[command.index("--n-candidates") + 1] == str(target.k)
    # The served name, not the registry key: that is what the endpoint answers to.
    assert command[command.index("--model") + 1] == BASE_MODELS[target.base].gateway_model


# --- the unattended driver ------------------------------------------------

def test_the_driver_judges_only_what_the_table_marks(monkeypatch):
    """The driver is the thing that will run for thirty hours with nobody
    watching, so it is the thing that must not judge by accident."""
    from gdanschin_runtime import run_sweep

    judged = []
    monkeypatch.setattr(run_sweep, "ensure_served", lambda t: True)
    monkeypatch.setattr(run_sweep, "generate", lambda t: None)
    monkeypatch.setattr(run_sweep, "judge_until",
                        lambda t, e: judged.append(t.base))

    for index, target in enumerate(oss_sweep.SWEEP, 1):
        run_sweep.run(target, index, len(oss_sweep.SWEEP))

    assert set(judged) == MEANT_TO_BE_JUDGED


def test_a_model_that_cannot_be_served_is_skipped_not_fatal(monkeypatch):
    """Thirty hours unattended: one model failing to start must not take the
    other eight with it."""
    from gdanschin_runtime import run_sweep

    generated = []
    monkeypatch.setattr(run_sweep, "ensure_served",
                        lambda t: t.base != oss_sweep.SWEEP[0].base)
    monkeypatch.setattr(run_sweep, "generate", lambda t: generated.append(t.base))
    monkeypatch.setattr(run_sweep, "judge_until", lambda t, e: None)

    assert run_sweep.main() == 0
    assert oss_sweep.SWEEP[0].base not in generated
    assert len(generated) == len(oss_sweep.SWEEP) - 1


def test_subprocesses_get_the_runtime_environment():
    """Without env.sh the model factory is unset and llm_monkeys builds a
    plain LiteLlm with the raw served name, which litellm rejects for having
    no provider - after the sweep has already started and looks fine."""
    from gdanschin_runtime import run_sweep

    wrapped = run_sweep.through_env(["python", "-m", "inference.cli"])
    assert wrapped[0] == "bash"
    assert "env.sh" in wrapped[2]
    assert wrapped[-3:] == ["python", "-m", "inference.cli"]


def test_generation_does_not_wait_for_judging(monkeypatch):
    """Judging works the external gateway and generation our own cards, so
    holding one for the other leaves the cards idle: on qwen3.5-4b at k=100
    generation took five hours and judging seventeen."""
    import threading
    from gdanschin_runtime import run_sweep

    order = []
    release = threading.Event()

    def slow_judge(target, generating):
        order.append(f"judge-start:{target.base}")
        release.wait(5)
        order.append(f"judge-end:{target.base}")

    monkeypatch.setattr(run_sweep, "ensure_served", lambda t: True)
    monkeypatch.setattr(run_sweep, "generate",
                        lambda t: order.append(f"generated:{t.base}"))
    monkeypatch.setattr(run_sweep, "judge_until", slow_judge)

    first = oss_sweep.SWEEP[0]
    thread = run_sweep.run(first, 1, 1)
    # Generation of the next model may start while this judge is still going.
    assert f"generated:{first.base}" in order
    assert f"judge-end:{first.base}" not in order
    release.set()
    thread.join()
    assert f"judge-end:{first.base}" in order
