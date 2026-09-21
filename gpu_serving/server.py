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

READY_TIMEOUT = 900       # weights load plus CUDA graph capture, worst case
FREE_TIMEOUT = 180        # how long the driver may take to report cards free
POLL = 3


class NoDriver(RuntimeError):
    """The GPU driver cannot be queried here."""


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
    # A stale file outlives a crashed server; treat a dead pid as nothing running.
    try:
        os.kill(state.pid, 0)
    except OSError:
        return None
    return state


def launch_argv(model: ServedModel, settings: Settings) -> list[str]:
    """The exact command that serves `model`. Pure, so it can be tested dry."""
    replicas = model.replicas(len(settings.cards))
    return [
        str(settings.python), "-m", "sglang_router.launch_server",
        "--model-path", str(settings.models_root / model.weights),
        "--served-model-name", model.served_name,
        "--tp-size", str(model.tp),
        "--dp-size", str(replicas),
        "--context-length", str(model.context_length),
        "--mem-fraction-static", str(model.mem_fraction),
        *model.sglang_args,
        "--host", settings.host,
        "--port", str(settings.port),
    ]


def start(name: str, settings: Settings | None = None) -> State:
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
    weights = settings.models_root / model.weights
    # Checked here rather than left to SGLang: a missing directory otherwise
    # surfaces minutes later as a traceback in a log nobody is tailing yet.
    if not weights.is_dir():
        raise FileNotFoundError(f"weights not on this box: {weights}")

    settings.run_dir.mkdir(parents=True, exist_ok=True)
    log = settings.run_dir / f"{name}.log"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(c) for c in settings.cards),
    }
    argv = launch_argv(model, settings)
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
        f"not ready after {timeout}s; last lines of {state.log}:\n{log_tail(state)}"
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
        if all(mb < 1024 for mb in used.values()):
            _state_path(settings).unlink(missing_ok=True)
            return
        time.sleep(POLL)

    raise TimeoutError(
        f"cards still busy {timeout}s after stopping {state.model}: "
        f"{cards_free(settings)}"
    )


__all__ = [
    "State", "start", "stop", "wait_ready", "read_state", "launch_argv",
    "log_tail", "cards_free", "NoDriver",
]
