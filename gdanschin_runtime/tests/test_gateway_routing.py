"""That each model is asked of the gateway that serves it.

The pool follows the provider a name resolves to, not the job the model is
doing. Anything the internal gateway serves is reached at its own provider
path; vendor models, and whatever only the external sglang serves, live on the
external one. That today's base models all happen to sit on the internal
gateway and today's judge is a vendor model is a fact about the current set,
not a rule - a vendor model can be a base model and an open-weight one can
judge.

Getting it wrong is not cosmetic. The internal gateway does not serve Gemini
at all, and the external one holds a per-user limit on requests in flight, so
a model that quietly changes pools either fails outright or spends the quota
another step is relying on.

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

# Where each name we run today is served. Kept as data rather than derived, so
# that a model moving pools has to be written down deliberately - which is the
# only way anyone would notice it moved.
# "local" means no gateway at all: the model is served from our own GPUs and
# the entry carries a base_url. It is written down here too, so that moving a
# model onto or off our hardware is as deliberate as moving it between pools.
EXPECTED_POOL = {
    "google/gemma-4-26B-A4B-it": "local",
    "openai/gpt-oss-120b": "local",
    "Qwen/Qwen3.8-27B-FP8": "local",
    "google/gemma-4-E2B-it": "local",
    "openai/gpt-oss-20b": "local",
    "Qwen/Qwen3.5-0.8B": "local",
    "Qwen/Qwen3.5-2B": "local",
    "Qwen/Qwen3.5-4B": "local",
    "Qwen/Qwen3.5-9B": "local",
    "Qwen/Qwen3.6-27B-FP8": "local",
    "Qwen/Qwen3.6-35B-A3B-FP8": "local",
    "Qwen/Qwen3.5-122B-A10B-FP8": "local",
    "MiniMaxAI/MiniMax-M2.5": "local",
    "gemma-4-26b-internal": "internal",
    "gpt-oss-120b": "internal",
    "deepseek-v4-flash": "internal",
    "qwen3.6-27b-noreasoning": "internal",
    "qwen3.8-27b-noreasoning": "internal",
    "gemini-3.8-flash": "external",
}


@pytest.fixture(autouse=True)
def stubbed_pools(monkeypatch):
    """Both gateways, without a configuration file or a token in sight."""
    asked: list[str] = []

    def load(kind: str = "external"):
        asked.append(kind)
        return POOLS[kind]

    monkeypatch.setattr(gateway, "load_gateway", load)
    return asked


def pool_of(model: str) -> str:
    """Which gateway this model would be asked of."""
    api_base = gateway.completion_kwargs(model)["api_base"]
    for kind, (url, _) in POOLS.items():
        if api_base.startswith(url):
            return kind
    raise AssertionError(f"neither pool: {api_base}")


@pytest.mark.parametrize("model,expected", sorted(
    (m, p) for m, p in EXPECTED_POOL.items() if p != "local"))
def test_each_model_is_served_by_the_pool_it_is_meant_to_be(model, expected):
    assert pool_of(model) == expected


@pytest.mark.parametrize("entry", list(BASE_MODELS.values()) + list(JUDGE_MODELS.values()),
                         ids=lambda e: e.name)
def test_every_model_in_the_registry_has_a_pool_written_down(entry):
    """A new entry has to say where it is served, rather than inherit whichever
    pool the neighbouring models happen to use."""
    assert entry.gateway_model in EXPECTED_POOL, (
        f"{entry.name} is not in EXPECTED_POOL: say which gateway serves it, "
        f"or 'local' if we serve it ourselves"
    )
    expected = EXPECTED_POOL[entry.gateway_model]

    # The two declarations have to agree. An entry that says "local" here but
    # carries no base_url would be sent to a gateway that does not serve it;
    # one with a base_url but a gateway pool here would spend a quota nobody
    # meant to spend.
    # getattr, because only base models can be served locally today; a judge
    # has no base_url field at all, and that is itself the answer.
    base_url = getattr(entry, "base_url", "")
    assert (expected == "local") == bool(base_url), (
        f"{entry.name}: EXPECTED_POOL says {expected!r} but base_url is "
        f"{base_url!r}"
    )
    if expected == "local":
        return
    assert pool_of(entry.gateway_model) == expected


def test_a_locally_served_model_never_reaches_a_gateway():
    """The whole point of the separate entry: our own GPUs, our own results
    directory, and no gateway quota spent. Routed on base_url, before any
    provider is inferred - the served name contains a slash, which would
    otherwise be read as "provider/model" and sent to the Gemini proxy."""
    from gdanschin_runtime.adapters import factory

    class Config:
        resolved_model_name = "google/gemma-4-26B-A4B-it"

    built = factory.build(Config())
    assert built.model == "openai/google/gemma-4-26B-A4B-it"
    assert built._additional_args["api_base"] == "http://127.0.0.1:8000/v1"


def test_the_pool_follows_the_provider_not_the_job():
    """A vendor model judging and a vendor model answering are the same call to
    the same gateway; so are an open-weight model in either role."""
    assert pool_of("gemini-3.8-flash") == "external"      # vendor, whatever it does
    assert pool_of("gpt-oss-120b") == "internal"          # open weights, likewise


def test_a_provider_the_gateway_does_not_list_is_taken_to_be_internal():
    """Open-weight models arrive on the internal gateway as their own provider,
    at /proxy/<provider>/v1, and the list of them is expected to grow."""
    kwargs = gateway.completion_kwargs("some-new-model/org/Some-New-Model")

    assert kwargs["api_base"] == "https://internal.test/proxy/some-new-model/v1"
    assert kwargs["model"] == "openai/org/Some-New-Model"


@pytest.mark.parametrize("model", ["openai/gpt-4o", "anthropic/claude-3", "xai/grok"])
def test_a_named_vendor_provider_is_external(model):
    assert pool_of(model) == "external"


@pytest.mark.parametrize("model", ["gemma-4-26b", "gemma-4-26b-a4b-it",
                                   "gemma-4-26b-internal", "qwen3-reranker-4b"])
def test_open_weight_aliases_are_internal(model):
    """The external gateway's sglang serves these too, but open-weight models
    are reached on the internal gateway, so a manual ask("gemma-4-26b") hits
    the same deployment the runs do rather than the external quota."""
    assert pool_of(model) == "internal"


def test_an_unlisted_gemma_is_not_guessed_onto_a_gateway():
    with pytest.raises(ValueError):
        gateway.resolve("gemma-9-99b")


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
