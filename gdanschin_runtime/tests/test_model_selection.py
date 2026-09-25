"""That a run calls the model it was asked for, and stores it where it says.

Both halves of that have been wrong here. --base once set the run name and
nothing else, so every step 1 run called the same default model and filed it
under the name that was asked for; --judge-name named a directory while the
verifier kept its own model and temperature. Neither showed up anywhere until
a finished run was read, and one of them survived a full overnight plan.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gdanschin_runtime import _bootstrap  # noqa: F401,E402  (puts llm_monkeys on the path)
from gdanschin_runtime.adapters.factory import gateway_name  # noqa: E402
from gdanschin_runtime.models import BASE_MODELS, JUDGE_MODELS, difficult_run  # noqa: E402
from gdanschin_runtime.run_step1 import build_config, create_parser  # noqa: E402

from config import resolve_model_name  # noqa: E402


@pytest.mark.parametrize("entry", list(BASE_MODELS.values()) + list(JUDGE_MODELS.values()),
                         ids=lambda e: e.name)
def test_a_registered_name_reaches_the_gateway_unchanged(entry):
    """The reference rewrites names it does not know into Vertex AI ones, which
    is how gpt-oss-120b became vertex_ai/openai/gpt-oss-120b-maas: an address
    the gateway does not serve, with nothing downstream able to tell."""
    assert gateway_name(resolve_model_name(entry.gateway_model)) == entry.gateway_model


@pytest.mark.parametrize("base", sorted(BASE_MODELS))
def test_base_picks_the_model_and_the_directory(base):
    config = build_config(create_parser().parse_args(["--base", base]))
    entry = BASE_MODELS[base]

    assert gateway_name(config.resolved_model_name) == entry.gateway_model
    assert config.resolved_run_name == difficult_run(base)
    assert config.max_tokens == entry.max_tokens


def test_two_models_do_not_share_a_directory():
    names = {build_config(create_parser().parse_args(["--base", base])).resolved_run_name
             for base in BASE_MODELS}
    assert len(names) == len(BASE_MODELS)


def test_an_unknown_base_is_refused_rather_than_defaulted():
    """The bug this file exists for: an unrecognised name used to fall back to
    the default model and still be stored under the name that was asked for."""
    from gdanschin_runtime.run_step1 import main

    with pytest.raises(KeyError):
        build_config(create_parser().parse_args(["--base", "no-such-model"]))
    assert main(["--base", "no-such-model"]) == 2


def test_max_tokens_can_still_be_overridden():
    config = build_config(create_parser().parse_args(
        ["--base", "gemma-4-26b", "--max-tokens", "77"]))
    assert config.max_tokens == 77
