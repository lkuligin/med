"""Serve one model at a time from the command line.

    ./gpu_serving/serve.sh list
    ./gpu_serving/serve.sh command gemma-4-26b     # print, run nothing
    ./gpu_serving/serve.sh up gemma-4-26b          # start and wait until ready
    ./gpu_serving/serve.sh status
    ./gpu_serving/serve.sh verify
    ./gpu_serving/serve.sh down

"up" does not return until the server can generate, so a script may follow it
immediately with a run. "down" does not return until the cards are free, so a
sweep may follow it immediately with the next model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from gpu_serving import check, fetch as fetch_module, host, logs, server
from gpu_serving.catalog import SERVABLE, describe
from gpu_serving.config import load


def _up(args: argparse.Namespace) -> int:
    settings = load()
    try:
        host.require_serving(settings)
    except host.CannotServe as error:
        print(error, file=sys.stderr)
        return 1
    state = server.start(args.model, settings, tuple(args.sglang_args))
    print(f"{state.model}: pid {state.pid}, {state.replicas} replicas on "
          f"cards {state.cards}, log {state.log}")
    print("waiting until it can generate ...")
    try:
        took = server.wait_ready(settings, timeout=args.timeout)
    except (TimeoutError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"ready in {took:.0f}s at {settings.base_url}")
    return 0 if args.no_verify else _verify(args)


def _verify(args: argparse.Namespace) -> int:
    settings = load()
    state = server.read_state(settings)
    if state is None:
        print("no server is running", file=sys.stderr)
        return 1
    result = check.verify(settings.base_url, state.served_name,
                          max_tokens=args.max_tokens)
    print(result.report())
    return 0 if result.ok else 1


def _down(args: argparse.Namespace) -> int:
    settings = load()
    state = server.read_state(settings)
    if state is None:
        print("nothing running")
        return 0
    print(f"stopping {state.model} (pid {state.pid}) ...")
    try:
        server.stop(settings)
    except TimeoutError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"stopped; cards free: {server.cards_free(settings)}")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    try:
        return fetch_module.fetch(args.model)
    except (PermissionError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1


def _status(args: argparse.Namespace) -> int:
    settings = load()
    capabilities = host.inspect(settings)
    if not capabilities.can_serve:
        print("this machine cannot serve:")
        for problem in capabilities.missing(settings):
            print(f"  - {problem}")
        print("list, command and stats still work here.")
        return 0
    state = server.read_state(settings)
    if state is None:
        print(f"nothing running (would serve at {settings.base_url})")
    else:
        age = time.time() - state.started_at
        print(f"{state.model} -> {state.served_name}")
        print(f"  pid {state.pid}, port {state.port}, {state.replicas} replicas")
        print(f"  cards {state.cards}, up {age / 60:.0f} min, log {state.log}")
    print(f"  memory in use per card: {server.cards_free(settings)}")
    return 0


def _list(args: argparse.Namespace) -> int:
    describe()
    return 0


def _command(args: argparse.Namespace) -> int:
    print(" ".join(server.launch_argv(SERVABLE[args.model], load(),
                                      tuple(args.sglang_args))))
    return 0


def _newest_log(args: argparse.Namespace) -> Path | None:
    """The log to read when none was named: the running one, else the latest."""
    settings = load()
    if args.log:
        return Path(args.log)
    state = server.read_state(settings)
    if state is not None:
        return Path(state.log)
    candidates = sorted(settings.run_dir.glob("*.log"),
                        key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _stats(args: argparse.Namespace) -> int:
    path = _newest_log(args)
    if path is None:
        print("no server log here; pass one: serve.sh stats <file>",
              file=sys.stderr)
        return 1
    if not path.is_file():
        print(f"no such log: {path}", file=sys.stderr)
        return 1
    print(f"{path}\n")
    print(logs.summarise(path.read_text(errors="replace")).render())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="serve.sh", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    up = sub.add_parser("up", help="start a model and wait until it can generate")
    up.add_argument("model", choices=sorted(SERVABLE))
    up.add_argument("--timeout", type=int, default=server.READY_TIMEOUT)
    up.add_argument("--max-tokens", type=int, default=256)
    up.add_argument("--no-verify", action="store_true")
    up.add_argument("sglang_args", nargs="*", metavar="-- ARG...",
                    help="extra flags passed to SGLang, overriding the catalogue")
    up.set_defaults(func=_up)

    verify = sub.add_parser("verify", help="check the running endpoint")
    verify.add_argument("--max-tokens", type=int, default=256)
    verify.set_defaults(func=_verify)

    for name, func, helptext in (
        ("down", _down, "stop, and wait for the cards to be released"),
        ("status", _status, "what is running"),
        ("list", _list, "what can be served"),
    ):
        sub.add_parser(name, help=helptext).set_defaults(func=func)

    fetch = sub.add_parser("fetch", help="download a model's weights (tens of GB)")
    fetch.add_argument("model", choices=sorted(SERVABLE))
    fetch.set_defaults(func=_fetch)

    stats = sub.add_parser("stats", help="summarise a server log; works without a GPU")
    stats.add_argument("log", nargs="?", help="defaults to the newest log here")
    stats.set_defaults(func=_stats)

    command = sub.add_parser("command", help="print the launch command, run nothing")
    command.add_argument("model", choices=sorted(SERVABLE))
    command.add_argument("sglang_args", nargs="*", metavar="-- ARG...")
    command.set_defaults(func=_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
