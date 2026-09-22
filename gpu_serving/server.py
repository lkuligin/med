"""Start one model server, wait until it can actually generate, and stop it.

    from gpu_serving import server

    server.start("gemma-4-26b")
    server.wait_ready()
    server.verify("gemma-4-26b")
    ...
    server.stop()

Three things here are less obvious than they look.

*Readiness is not liveness.* SGLang captures CUDA graphs during startup, so by
the time it answers at all it is already warm - no warm-up loop is needed. But
/health answers before the model can generate; /health_generate answers only
once it can. Polling the wrong one starts the run against a server that then
stalls on the first request.

*A detached server has to be findable again.* The process is started with
setsid so it outlives the shell (and so it inherits the launching shell's
environment, which a tmux session would not). That means the next invocation
cannot rely on being its parent, hence the state file.

*Stopping is not done when the process exits.* With replicas there are several
worker processes, and a card is only free when the driver says so. Starting the
next model before then fails with an out-of-memory error that names nothing.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from gpu_serving.catalog import SERVABLE, ServedModel
from gpu_serving.config import Settings, load

# Weights, CUDA graph capture, and - for an FP8 checkpoint - DeepGEMM
# warming its kernels. That last one is the reason this is not minutes:
# Qwen3.8-27B-FP8 was still warming at twelve minutes, where the bf16 Gemma
# was ready in 87 seconds and MXFP4 gpt-oss in 151. Timing out early does not
# stop the server, which is detached and keeps coming up; it just reports a
# failure that is not one, and teaches whoever sees it to ignore the check.
READY_TIMEOUT = 2400
FREE_TIMEOUT = 180        # how long the driver may take to report cards free
POLL = 3

# The module the server runs as, and the fingerprint used to recognise its
# process later. Named once so the command and the check cannot drift apart.
LAUNCHER = "sglang.launch_server"


class NoDriver(RuntimeError):
    """The GPU driver cannot be queried here."""


class NoEnvironment(RuntimeError):
    """The environment this model launches from has not been built here."""


@dataclass(frozen=True)
class State:
    """What is running, written where the next invocation can find it."""

    model: str
    served_name: str
    pid: int
    port: int
    cards: list[int]
    replicas: int
    log: str
    started_at: float


def _state_path(settings: Settings) -> Path:
    return settings.run_dir / "server.json"


def read_state(settings: Settings | None = None) -> State | None:
    """The server this package believes is running, or None."""
    settings = settings or load()
    path = _state_path(settings)
    if not path.is_file():
        return None
    state = State(**json.loads(path.read_text()))
    # A stale file outlives a crashed server, so the pid it names has to be
    # checked - and checked properly. os.kill(pid, 0) is not enough: it
    # succeeds for a zombie, whose process has exited and is only waiting to be
    # reaped, and it succeeds for whatever unrelated program later inherits
    # that number. Both would have us report a server that is not there.
    return state if _alive(state.pid) else None


def _alive(pid: int) -> bool:
    """Is this pid a live server of ours?"""
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            # Everything after the executable name, which itself may contain
            # spaces and is parenthesised; the state letter is the first field.
            state_letter = stat.read_text().rsplit(") ", 1)[-1].split()[0]
            if state_letter == "Z":
                return False
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (OSError, IndexError):
            return False
        return LAUNCHER.encode() in cmdline.replace(b"\0", b" ")

    # No /proc: not Linux, so not a box we serve on. Fall back to the weaker
    # check rather than claiming to know.
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def launch_argv(model: ServedModel, settings: Settings,
                extra: tuple[str, ...] = ()) -> list[str]:
    """The exact command that serves `model`. Pure, so it can be tested dry.

    `extra` is appended verbatim, so a flag given there overrides the same
    flag from the catalogue.

    Data parallelism comes from SGLang's own --dp-size rather than the separate
    sglang-router package. The router would add prefix-cache-aware dispatch,
    which suits repeated sampling of one question, but its latest release
    (0.3.2) imports a symbol sglang 0.5.19 no longer exports and cannot start
    at all. Worth revisiting when the package catches up.
    """
    replicas = model.replicas(len(settings.cards))
    return [
        str(settings.python_for(model.venv)), "-m", LAUNCHER,
        "--model-path", resolve_weights(model, settings),
        "--served-model-name", model.served_name,
        "--tp-size", str(model.tp),
        "--dp-size", str(replicas),
        "--context-length", str(model.context_length),
        "--mem-fraction-static", str(model.mem_fraction),
        *model.sglang_args,
        "--host", settings.host,
        "--port", str(settings.port),
        # Last, so an override on the command line beats the catalogue: SGLang
        # takes the final occurrence of a repeated flag. That is how a backend
        # or a budget gets tried before it is written down.
        *extra,
    ]


def resolve_weights(model: ServedModel, settings: Settings) -> str:
    """What to pass as --model-path: a directory if we have one, else the id.

    Two shapes exist on this box and both are needed. /mnt/data/models holds
    flat <org>/<name> directories but belongs to another user, so nothing new
    can go there. Anything we fetch ourselves lands in a Hugging Face cache,
    whose layout is models--org--name/snapshots/<hash> - a path nobody should
    have to write down. So a model not found in any flat root is passed to
    SGLang as its repo id, and HF_HOME in the environment is what turns that
    into weights.
    """
    for root in settings.models_roots:
        candidate = root / model.weights
        if candidate.is_dir():
            return str(candidate)
    return model.weights


def cached_in_hf_home(model: ServedModel, settings: Settings) -> bool:
    """Is this model already in the Hugging Face cache we point SGLang at?"""
    folder = "models--" + model.weights.replace("/", "--")
    return (settings.hf_home / "hub" / folder).is_dir()


def server_env(settings: Settings, venv: str = "") -> dict[str, str]:
    """The environment the server runs in. Pure, so a test can read it.

    Running .venv/bin/python directly is not the same as activating the venv:
    the interpreter is right, but the venv's bin directory is not on PATH, so
    any tool it provides as a console script is invisible. SGLang shells out to
    ninja while capturing CUDA graphs, which made this surface as
    FileNotFoundError several minutes into a launch that had already loaded the
    weights and started every replica.

    `venv` is the catalogue entry's, so a model served from the second
    environment gets that one's bin directory rather than the default's - the
    two hold different SGLangs, and a PATH pointing at the wrong one is the
    same failure as above with a more confusing cause.
    """
    chosen = settings.venv_for(venv)
    bin_dir = str(chosen / "bin")
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(c) for c in settings.cards),
        "VIRTUAL_ENV": str(chosen),
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        # How a repo id passed as --model-path becomes weights. Offline on
        # purpose: serving is not the moment to discover that a download is
        # needed, and "serve.sh fetch" is where that belongs.
        "HF_HOME": str(settings.hf_home),
        # All three, because SGLang, its JIT layer and DeepGEMM each read
        # their own. Leaving any of them unset puts part of the cache
        # somewhere the rest is not.
        "SGLANG_CACHE_DIR": str(settings.kernel_cache),
        "DG_CACHE_HOME": str(settings.kernel_cache),
        "DG_JIT_CACHE_DIR": str(settings.kernel_cache / "deep_gemm"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def start(name: str, settings: Settings | None = None,
          extra: tuple[str, ...] = ()) -> State:
    """Launch a model, detached, and record where it went."""
    settings = settings or load()
    if name not in SERVABLE:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(SERVABLE)}")
    running = read_state(settings)
    if running is not None:
        raise RuntimeError(
            f"{running.model} is already serving on port {running.port} "
            f"(pid {running.pid}); stop it first"
        )

    model = SERVABLE[name]
    # Checked before the weights, because building an environment is the
    # longer errand of the two and there is no point reporting them one at a
    # time. The flag is the one setup_env.sh takes, so the message is the fix.
    python = settings.python_for(model.venv)
    if not python.is_file():
        raise NoEnvironment(
            f"{name} is served from the {model.venv or 'default'} environment, "
            f"which is not built at {python.parent.parent}.\n"
            f"Build it: ./gpu_serving/setup_env.sh"
            + (f" --{model.venv}" if model.venv else ""))
    # Checked here rather than left to SGLang: missing weights otherwise
    # surface minutes later as a traceback in a log nobody is tailing yet.
    path = resolve_weights(model, settings)
    if not Path(path).is_dir() and not cached_in_hf_home(model, settings):
        roots = ", ".join(str(r) for r in settings.models_roots)
        raise FileNotFoundError(
            f"{model.weights} is not in any weights root ({roots}) nor in the "
            f"cache at {settings.hf_home}.\nFetch it: "
            f"./gpu_serving/serve.sh fetch {name}")

    settings.run_dir.mkdir(parents=True, exist_ok=True)
    log = settings.run_dir / f"{name}.log"
    env = server_env(settings, model.venv)
    argv = launch_argv(model, settings, extra)
    with log.open("ab") as handle:
        handle.write(f"\n=== {time.strftime('%F %T')} {' '.join(argv)}\n".encode())
        handle.flush()
        process = subprocess.Popen(
            argv, env=env, stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, cwd=settings.run_dir,
        )

    state = State(
        model=name,
        served_name=model.served_name,
        pid=process.pid,
        port=settings.port,
        cards=list(settings.cards),
        replicas=model.replicas(len(settings.cards)),
        log=str(log),
        started_at=time.time(),
    )
    _state_path(settings).write_text(json.dumps(asdict(state), indent=2))
    return state


def _get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, ""


def log_tail(state: State, lines: int = 40) -> str:
    """The end of the server's log - where the reason for a failure lives."""
    path = Path(state.log)
    if not path.is_file():
        return f"(no log at {path})"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def wait_ready(settings: Settings | None = None, timeout: int = READY_TIMEOUT) -> float:
    """Block until the server can generate. Returns how long that took."""
    settings = settings or load()
    state = read_state(settings)
    if state is None:
        raise RuntimeError("no server is running")

    deadline = time.time() + timeout
    while time.time() < deadline:
        # /health_generate runs a real generation; /health only proves the
        # process is up, which it is long before the weights are loaded.
        status, _ = _get(f"{settings.base_url}/health_generate", timeout=10)
        if status == 200:
            return time.time() - state.started_at
        try:
            os.kill(state.pid, 0)
        except OSError:
            raise RuntimeError(
                f"server died during startup; last lines of {state.log}:\n"
                f"{log_tail(state)}"
            ) from None
        time.sleep(POLL)

    raise TimeoutError(
        f"not ready after {timeout}s. The server is detached and may still be "
        f"coming up - check with 'serve.sh verify' before restarting anything. "
        f"Last lines of {state.log}:\n{log_tail(state)}"
    )


def cards_free(settings: Settings | None = None) -> dict[int, int]:
    """Megabytes still in use on our cards, per card, as the driver reports it."""
    settings = settings or load()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        # Raised rather than shrugged off: "down" uses this to decide the next
        # model may start, and a box where we cannot see the cards is a box
        # where that decision cannot be made.
        raise NoDriver("nvidia-smi not found: this is not a GPU box") from None
    used = {}
    for line in result.stdout.strip().splitlines():
        index, memory = (part.strip() for part in line.split(","))
        if int(index) in settings.cards:
            used[int(index)] = int(memory)
    return used


def lingering() -> list[int]:
    """Our SGLang processes that are still about after a stop.

    Free cards are not the whole story. A scheduler that has released its
    memory can still hold the distributed port SGLang picked, and the next
    server then dies with EADDRINUSE - which cost a model forty minutes of
    retries and a skip before this check existed. Waiting for the processes
    to go covers the ports without having to guess which ones they were.
    """
    result = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "sglang"],
                            capture_output=True, text=True)
    mine = os.getpid()
    return [int(line) for line in result.stdout.split()
            if line.isdigit() and int(line) != mine]


def stop(settings: Settings | None = None, timeout: int = FREE_TIMEOUT) -> None:
    """Stop the server and do not return until the cards are actually free."""
    settings = settings or load()
    state = read_state(settings)
    if state is None:
        _state_path(settings).unlink(missing_ok=True)
        return

    # The whole process group: start() used start_new_session, so the router
    # and every replica share the leader's pid as their group id.
    try:
        os.killpg(state.pid, signal.SIGINT)
    except OSError:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(state.pid, 0)
        except OSError:
            break
        time.sleep(POLL)
    else:
        os.killpg(state.pid, signal.SIGKILL)

    # Process exit is not the same as memory release. The next model will fail
    # to allocate if we start it while the driver still holds these pages.
    deadline = time.time() + timeout
    while time.time() < deadline:
        used = cards_free(settings)
        if all(mb < 1024 for mb in used.values()) and not lingering():
            _state_path(settings).unlink(missing_ok=True)
            return
        time.sleep(POLL)

    raise TimeoutError(
        f"{timeout}s after stopping {state.model}: cards "
        f"{cards_free(settings)}, lingering sglang processes {lingering()}"
    )


__all__ = [
    "State", "start", "stop", "wait_ready", "read_state", "launch_argv",
    "log_tail", "cards_free", "NoDriver", "LAUNCHER", "server_env",
    "lingering",
    "resolve_weights", "cached_in_hf_home",
]
