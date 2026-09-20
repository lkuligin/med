"""That each model is asked of the pool that serves it.

The two gateways are not interchangeable. Generation goes to the internal one,
which showed no rate limit and saturates around eight concurrent; judging goes
to the external one, where Gemini lives behind a per-user limit on requests in
flight. Sending step 2 to the external gateway spends the quota the judge
needs, and the internal one does not serve Gemini at all - so a model that
quietly moves pools either fails outright or starves the other step.

No credentials are read here: the pools are stubbed, and which one was asked
for is what the tests assert.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gdanschin_runtime import gateway  # noqa: E402
from gdanschin_runtime.models import BASE_MODELS, JUDGE_MODELS  # noqa: E402

POOLS = {"internal": ("https://internal.test", "token-internal"),
         "external": ("https://external.test", "token-external")}


@pytest.fixture(autouse=True)
def stubbed_pools(monkeypatch):
    """Both gateways, without a configuration file or a token in sight."""
    asked: list[str] = []

    def load(kind: str = "external"):
        asked.append(kind)
        return POOLS[kind]

    monkeypatch.setattr(gateway, "load_gateway", load)
    return asked


def pool_of(kwargs: dict) -> str:
    """Which gateway a set of completion arguments points at."""
    for kind, (url, _) in POOLS.items():
        if kwargs["api_base"].startswith(url):
            return kind
    raise AssertionError(f"neither pool: {kwargs['api_base']}")


@pytest.mark.parametrize("entry", list(BASE_MODELS.values()), ids=lambda e: e.name)
def test_every_base_model_is_generated_on_the_internal_pool(entry):
    assert pool_of(gateway.completion_kwargs(entry.gateway_model)) == "internal"


@pytest.mark.parametrize("entry", list(JUDGE_MODELS.values()), ids=lambda e: e.name)
def test_every_judge_is_asked_of_the_external_pool(entry):
    """Gemini is only served there, and its limit is the reason judging is kept
    apart from generation in the first place."""
    assert pool_of(gateway.completion_kwargs(entry.gateway_model)) == "external"


def test_the_two_gemmas_are_different_models_on_different_pools():
    """gemma-4-26b is served externally through sglang and
    gemma-4-26b-internal by the internal gateway. models.py names the second
    on purpose: the alias without the suffix would move generation onto the
    judge's quota without changing a single visible name."""
    external = gateway.completion_kwargs("gemma-4-26b")
    internal = gateway.completion_kwargs("gemma-4-26b-internal")

    assert pool_of(external) == "external"
    assert pool_of(internal) == "internal"
    assert BASE_MODELS["gemma-4-26b"].gateway_model == "gemma-4-26b-internal"


def test_an_internal_model_is_reached_at_its_own_provider_path():
    kwargs = gateway.completion_kwargs("gpt-oss-120b")

    assert kwargs["model"] == "openai/openai/gpt-oss-120b"
    assert kwargs["api_base"] == "https://internal.test/proxy/gpt-oss-120b/v1"
    assert kwargs["api_key"] == "token-internal"


def test_gemini_carries_its_key_in_a_header():
    """Without this the key travels as ?key=, which the gateway ignores - and
    the failure reads as a model problem rather than an unauthenticated call."""
    kwargs = gateway.completion_kwargs("gemini-3.8-flash")

    assert kwargs["model"].startswith("gemini/")
    assert kwargs["extra_headers"] == {"Authorization": "Bearer token-external"}


def test_a_pool_is_only_asked_for_once_per_call(stubbed_pools):
    gateway.completion_kwargs("qwen3.8-27b-noreasoning")
    assert stubbed_pools == ["internal"]


@pytest.mark.parametrize("model", ["qwen3.6-27b-noreasoning", "qwen3.8-27b-noreasoning"])
def test_thinking_is_turned_off_where_the_name_says_so(model):
    """Asked for medical facts with thinking on, Qwen3 spent the whole budget
    reasoning and returned an empty string with no error. The switch has to
    ride on the request; nothing else turns it off."""
    kwargs = gateway.completion_kwargs(model)

    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_a_model_no_pool_claims_is_refused():
    with pytest.raises(ValueError, match="cannot tell which provider"):
        gateway.completion_kwargs("some-model-nobody-serves")
