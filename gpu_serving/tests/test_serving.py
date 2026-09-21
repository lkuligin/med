"""Everything here runs without a GPU, an SGLang install or a network.

The point of the split is that the risky parts - which cards, which flags,
which model path - are pure functions, so they can be checked on a laptop
before anything is launched on a shared machine.
"""

from __future__ import annotations

import pytest

from gpu_serving.catalog import SERVABLE, ServedModel
from gpu_serving.check import LEAKED
from gpu_serving.config import Settings
from gpu_serving import host, logs
from gpu_serving import server as server_module
from gpu_serving.server import launch_argv


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        cards=(4, 5, 6, 7),
        models_root=tmp_path / "models",
        venv=tmp_path / "venv",
        host="127.0.0.1",
        port=8000,
        run_dir=tmp_path / "run",
    )


def test_every_catalogue_entry_fits_one_card():
    # The whole dp-over-tp argument rests on this; if a checkpoint grows past a
    # card the entry needs a tp, and the test should say so rather than the
    # server failing to allocate at three in the morning.
    for model in SERVABLE.values():
        assert model.weights_gb < 141, model.name


def test_replicas_use_every_card(settings):
    for model in SERVABLE.values():
        assert model.replicas(len(settings.cards)) * model.tp == len(settings.cards)


def test_tp_must_divide_the_cards():
    awkward = ServedModel(name="x", weights="w", served_name="s",
                          weights_gb=1, tp=3)
    with pytest.raises(ValueError, match="does not divide"):
        awkward.replicas(4)


def test_launch_command_serves_the_expected_name(settings):
    argv = launch_argv(SERVABLE["gemma-4-26b"], settings)
    assert argv[argv.index("--served-model-name") + 1] == "google/gemma-4-26B-A4B-it"
    assert argv[argv.index("--dp-size") + 1] == "4"
    assert argv[argv.index("--tp-size") + 1] == "1"


def test_launch_command_stays_on_loopback(settings):
    # A shared box: binding 0.0.0.0 would hand the model to the whole network.
    for model in SERVABLE.values():
        argv = launch_argv(model, settings)
        assert argv[argv.index("--host") + 1] == "127.0.0.1"


def test_launch_command_uses_the_configured_weights_root(settings):
    argv = launch_argv(SERVABLE["gpt-oss-120b"], settings)
    path = argv[argv.index("--model-path") + 1]
    assert path == str(settings.models_root / "openai/gpt-oss-120b")


def test_reasoning_leak_detection():
    assert LEAKED.search("<think>weighing the options</think> answer")
    assert LEAKED.search("<|channel|>analysis")
    assert not LEAKED.search("The answer is D, because the patient is febrile.")


@pytest.mark.parametrize("name", sorted(SERVABLE))
def test_reasoning_models_declare_a_parser(name):
    # Without a --reasoning-parser the reasoning arrives inside content, and
    # every local answer is then structurally unlike the gateway's.
    model = SERVABLE[name]
    if name == "gemma-4-26b":
        pytest.skip("no reasoning mode in this configuration")
    assert "--reasoning-parser" in model.sglang_args


def test_cards_free_reports_a_box_without_a_driver(monkeypatch, settings):
    # "status" is run on the laptop too, and a missing nvidia-smi must read as
    # "not a GPU box" rather than as a FileNotFoundError traceback.
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(server_module.subprocess, "run", missing)
    with pytest.raises(server_module.NoDriver, match="not a GPU box"):
        server_module.cards_free(settings)


# --- what a machine can do -------------------------------------------------

def test_a_laptop_is_told_why_it_cannot_serve(settings, monkeypatch):
    monkeypatch.setattr(host.shutil, "which", lambda _: None)
    capabilities = host.inspect(settings)
    assert not capabilities.can_serve
    problems = capabilities.missing(settings)
    assert any("no NVIDIA driver" in p for p in problems)
    # The message has to name the fix, not just the fault.
    assert any("setup_env.sh" in p for p in problems)


def test_require_serving_raises_something_readable(settings, monkeypatch):
    monkeypatch.setattr(host.shutil, "which", lambda _: None)
    with pytest.raises(host.CannotServe) as raised:
        host.require_serving(settings)
    assert "list, command, stats" in str(raised.value)


# --- reading a log ---------------------------------------------------------

BUSY = """
[t] The server is fired up and ready to roll!
[t] Prefill batch. #new-seq: 4, #new-token: 8192, #cached-token: 4096, cache hit rate: 33.30%, #queue-req: 0
[t] Decode batch. #running-req: 48, token usage: 0.31, gen throughput (token/s): 4210.55, #queue-req: 12
[t] Decode batch. #running-req: 52, token usage: 0.34, gen throughput (token/s): 4550.10, #queue-req: 7
"""

IDLE = """
[t] The server is fired up and ready to roll!
[t] Decode batch. #running-req: 2, token usage: 0.01, gen throughput (token/s): 95.20, #queue-req: 0
[t] Decode batch. #running-req: 1, token usage: 0.01, gen throughput (token/s): 48.10, #queue-req: 0
"""


def test_a_busy_log_reads_as_the_server_being_the_limit():
    summary = logs.summarise(BUSY)
    assert summary.ready
    assert summary.decode.count == 2
    assert max(summary.decode.get("#running-req")) == 52
    assert "server was the limit" in summary.verdict()


def test_an_idle_log_blames_the_client():
    # The reading that matters for dp4: four replicas fed two requests are
    # measuring the runner, and the throughput number means nothing.
    summary = logs.summarise(IDLE)
    assert "raise the run's concurrency" in summary.verdict()


def test_unknown_fields_do_not_break_the_parser():
    # Field sets differ by version and by what is enabled; a new label should
    # cost nothing.
    summary = logs.summarise(
        "[t] Decode batch. #running-req: 3, est. read BW (GB/s per GPU): 812.5, "
        "future-field: 7, #queue-req: 0\n")
    assert summary.decode.get("est. read BW (GB/s per GPU)") == [812.5]
    assert summary.decode.get("future-field") == [7.0]


def test_a_log_with_no_decoding_says_so():
    assert "nothing was generated" in logs.summarise("[t] starting up\n").render()
