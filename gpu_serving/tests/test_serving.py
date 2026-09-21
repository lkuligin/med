"""Everything here runs without a GPU, an SGLang install or a network.

The point of the split is that the risky parts - which cards, which flags,
which model path - are pure functions, so they can be checked on a laptop
before anything is launched on a shared machine.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from gpu_serving.catalog import SERVABLE, ServedModel
from gpu_serving import check as check_module
from gpu_serving.check import LEAKED
from gpu_serving.config import Settings
from gpu_serving import host, logs
from gpu_serving import server as server_module
from gpu_serving.server import launch_argv


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        cards=(4, 5, 6, 7),
        models_roots=(tmp_path / "shared", tmp_path / "ours"),
        hf_home=tmp_path / "hf",
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
    (settings.models_roots[0] / "openai/gpt-oss-120b").mkdir(parents=True)
    argv = launch_argv(SERVABLE["gpt-oss-120b"], settings)
    path = argv[argv.index("--model-path") + 1]
    assert path == str(settings.models_roots[0] / "openai/gpt-oss-120b")


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


# --- is that pid really our server? ---------------------------------------

def _write_state(settings, pid):
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    (settings.run_dir / "server.json").write_text(json.dumps({
        "model": "gemma-4-26b", "served_name": "google/gemma-4-26B-A4B-it",
        "pid": pid, "port": 8000, "cards": [4, 5, 6, 7], "replicas": 4,
        "log": "/dev/null", "started_at": 0.0}))


def test_a_zombie_is_not_a_running_server(settings, monkeypatch, tmp_path):
    # os.kill(pid, 0) succeeds for a process that has exited and is waiting to
    # be reaped. Believing it left "up" refusing to start over a dead server.
    proc = tmp_path / "proc" / "999"
    proc.mkdir(parents=True)
    (proc / "stat").write_text("999 (python) Z 1 999 0 0 -1 0 0 0\n")
    (proc / "cmdline").write_bytes(b"python\0-m\0sglang.launch_server\0")
    monkeypatch.setattr(server_module, "Path",
                        lambda p: tmp_path / p.lstrip("/") if str(p).startswith("/proc")
                        else pathlib.Path(p))
    _write_state(settings, 999)
    assert server_module.read_state(settings) is None


def test_a_recycled_pid_is_not_our_server(settings, monkeypatch, tmp_path):
    # The number outlived the server and now belongs to something else.
    proc = tmp_path / "proc" / "999"
    proc.mkdir(parents=True)
    (proc / "stat").write_text("999 (vim) S 1 999 0 0 -1 0 0 0\n")
    (proc / "cmdline").write_bytes(b"vim\0notes.txt\0")
    monkeypatch.setattr(server_module, "Path",
                        lambda p: tmp_path / p.lstrip("/") if str(p).startswith("/proc")
                        else pathlib.Path(p))
    _write_state(settings, 999)
    assert server_module.read_state(settings) is None


def test_our_own_live_server_is_recognised(settings, monkeypatch, tmp_path):
    proc = tmp_path / "proc" / "999"
    proc.mkdir(parents=True)
    (proc / "stat").write_text("999 (python) S 1 999 0 0 -1 0 0 0\n")
    (proc / "cmdline").write_bytes(b"python\0-m\0sglang.launch_server\0--dp-size\0004\0")
    monkeypatch.setattr(server_module, "Path",
                        lambda p: tmp_path / p.lstrip("/") if str(p).startswith("/proc")
                        else pathlib.Path(p))
    _write_state(settings, 999)
    state = server_module.read_state(settings)
    assert state is not None and state.model == "gemma-4-26b"


def test_the_server_can_find_its_own_tools(settings):
    # SGLang shells out to ninja from the venv while capturing CUDA graphs.
    # Running .venv/bin/python does not put .venv/bin on PATH, so without this
    # the launch dies at the last step, minutes in.
    env = server_module.server_env(settings)
    assert env["PATH"].startswith(str(settings.venv / "bin") + ":")
    assert env["VIRTUAL_ENV"] == str(settings.venv)
    assert env["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"


def test_an_override_comes_after_the_catalogue(settings):
    # SGLang takes the last occurrence of a repeated flag, so appending is what
    # makes "try this backend without editing the catalogue" work.
    argv = server_module.launch_argv(SERVABLE["gemma-4-26b"], settings,
                                     ("--attention-backend", "flashinfer"))
    assert argv[-2:] == ["--attention-backend", "flashinfer"]


# --- would verification have caught the broken backend? -------------------

def test_pad_spam_is_degenerate():
    # Exactly what trtllm_mha produced for Gemma 4: a server that loads, serves
    # and answers, while every answer is padding.
    assert "special token" in check_module.degenerate("<pad>" * 40)


def test_one_word_forever_is_degenerate():
    assert "of the words are" in check_module.degenerate("the " * 60)


def test_real_prose_is_not_degenerate():
    text = ("The vagus nerve carries parasympathetic fibres to the thorax and "
            "abdomen, and its injury causes hoarseness. The facial nerve "
            "supplies the muscles of expression and taste to the anterior "
            "tongue. The optic nerve carries vision from the retina.")
    assert check_module.degenerate(text) is None


def test_a_short_reply_is_never_called_degenerate():
    # Under 40 words the repetition test would fire on ordinary answers.
    assert check_module.degenerate("PONG") is None
    assert check_module.degenerate("The answer is D. The answer is D.") is None


# --- where the weights come from -----------------------------------------

def test_a_later_root_is_used_when_the_shared_one_lacks_the_model(settings):
    # The shared directory belongs to another user and is read-only, so
    # anything we add ourselves has to be found somewhere else.
    model = SERVABLE["qwen3.8-27b"]
    (settings.models_roots[1] / model.weights).mkdir(parents=True)
    assert server_module.resolve_weights(model, settings) == str(
        settings.models_roots[1] / model.weights)


def test_an_unfound_model_is_passed_as_a_repo_id(settings):
    # Nothing on disk in a flat layout: SGLang gets the id and HF_HOME turns
    # it into weights, so nobody has to write out a cache snapshot path.
    model = SERVABLE["qwen3.8-27b"]
    assert server_module.resolve_weights(model, settings) == model.weights


def test_the_server_is_pointed_at_the_cache_and_kept_offline(settings):
    env = server_module.server_env(settings)
    assert env["HF_HOME"] == str(settings.hf_home)
    # Starting a server is not the moment to discover a download is needed.
    assert env["HF_HUB_OFFLINE"] == "1"


def test_fetching_is_online_and_skips_duplicate_weights(settings):
    from gpu_serving import fetch as fetch_module

    env = fetch_module.fetch_env(settings)
    assert env["HF_HUB_OFFLINE"] == "0"
    assert env["HF_HOME"] == str(settings.hf_home)

    argv = fetch_module.fetch_argv(SERVABLE["gpt-oss-120b"], settings)
    # gpt-oss ships the same weights three ways; without this the download is
    # twice the size for nothing.
    assert "original/*" in argv and "metal/*" in argv
