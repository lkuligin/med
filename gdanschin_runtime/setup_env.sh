#!/usr/bin/env bash
# Build the Python environment. Same command on this laptop and on the GPU box.
#
#   ./gdanschin_runtime/setup_env.sh --cpu          this laptop: pipeline only
#   ./gdanschin_runtime/setup_env.sh --gpu           GPU box: also the serving stack
#   ./gdanschin_runtime/setup_env.sh --gpu --recreate
#   ./gdanschin_runtime/setup_env.sh --verify-only
#   ./gdanschin_runtime/setup_env.sh --gpu --quiet     suppress pip progress
#
# Exactly one of --cpu / --gpu is required rather than detected, because the
# expensive mistake is silent: build a CPU environment on the GPU box and
# nothing complains until a serving run fails hours later. The script still
# cross-checks the choice against nvidia-smi and refuses an obvious mismatch.
#
# No sudo, ever: we have no root on the GPU box. Where Ubuntu ships python3
# without ensurepip (its venv support lives in a separate apt package), the
# script creates the venv with --without-pip and bootstraps pip from a wheel
# fetched off PyPI. Note it cannot use bootstrap.pypa.io/get-pip.py: that host
# is blocked from the GPU box, while pypi.org is reachable.
#
# .venv/ is excluded from the mirror and from git, so each machine builds its
# own against its own platform - which is the point, since the GPU box needs
# CUDA wheels this laptop must never see.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
REQ="$REPO/llm_monkeys/requirements.txt"
REQ_DEV="$REPO/gdanschin_runtime/requirements-dev.txt"
REQ_GPU="$REPO/gdanschin_runtime/requirements-gpu.txt"

RECREATE=0; VERIFY_ONLY=0; QUIET=0; PROFILE=""
for arg in "$@"; do
    case "$arg" in
        --cpu)         PROFILE="cpu" ;;
        --gpu)         PROFILE="gpu" ;;
        --recreate)    RECREATE=1 ;;
        --verify-only) VERIFY_ONLY=1 ;;
        --quiet)       QUIET=1 ;;
        -h|--help)     sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

has_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

if [[ $VERIFY_ONLY -eq 0 ]]; then
    if [[ -z "$PROFILE" ]]; then
        echo "say which environment this is: --cpu or --gpu" >&2
        if has_gpu; then
            echo "  nvidia-smi reports $(nvidia-smi -L | wc -l | tr -d ' ') GPU(s) here, so you probably want --gpu" >&2
        else
            echo "  no GPU detected here, so you probably want --cpu" >&2
        fi
        exit 2
    fi
    if [[ "$PROFILE" == "gpu" ]] && ! has_gpu; then
        echo "--gpu was given but no GPU is visible (nvidia-smi missing or reports none)." >&2
        echo "Use --cpu, or check you are on the right machine." >&2
        exit 1
    fi
    if [[ "$PROFILE" == "cpu" ]] && has_gpu; then
        echo "note: GPUs are present but --cpu was given; installing the pipeline only."
    fi
fi

# --- locate an interpreter the project supports -----------------------------
# >= 3.11 for typing.Self (llm_monkeys/config.py), < 3.15 for litellm. 3.12 is
# what the GPU box ships, so prefer it and keep both machines on one version.
find_python() {
    for c in /opt/homebrew/bin/python3.12 python3.12 python3 python3.13 python3.11; do
        command -v "$c" >/dev/null 2>&1 || continue
        "$c" -c 'import sys; raise SystemExit(0 if (3,11) <= sys.version_info < (3,15) else 1)' 2>/dev/null \
            && { command -v "$c"; return 0; }
    done
    return 1
}

PY="$(find_python)" || {
    echo "no supported Python found (need >= 3.11, < 3.15; 3.12 preferred)" >&2
    [[ "$(uname -s)" == "Darwin" ]] && echo "  brew install python@3.12" >&2
    exit 1
}
echo "interpreter: $PY ($("$PY" -V 2>&1))"

# Fetch the newest pure-Python pip wheel and install it into the venv. A wheel
# is a zip, and pip's own wheel is runnable in place, which is what breaks the
# chicken-and-egg of needing pip to install pip.
bootstrap_pip() {
    local tmp whl
    tmp="$(mktemp -d)"
    echo "ensurepip is unavailable; bootstrapping pip from PyPI"
    "$PY" - "$tmp" <<'PYEOF'
import json, sys, urllib.request
from pathlib import Path
out = Path(sys.argv[1])
data = json.load(urllib.request.urlopen("https://pypi.org/pypi/pip/json", timeout=30))
wheel = next(u for u in data["urls"] if u["filename"].endswith("-py3-none-any.whl"))
(out / wheel["filename"]).write_bytes(urllib.request.urlopen(wheel["url"], timeout=180).read())
print(f"  fetched {wheel['filename']}")
PYEOF
    whl="$(ls "$tmp"/pip-*.whl | head -1)"
    "$VENV/bin/python" "$whl/pip" install --quiet --no-index --find-links "$tmp" pip
    rm -rf "$tmp"
}

if [[ $VERIFY_ONLY -eq 0 ]]; then
    [[ $RECREATE -eq 1 ]] && { echo "removing $VENV"; rm -rf "$VENV"; }

    if [[ -d "$VENV" ]]; then
        echo "reusing existing $VENV"
    elif "$PY" -c 'import ensurepip' 2>/dev/null; then
        echo "creating $VENV"
        "$PY" -m venv "$VENV"
    else
        echo "creating $VENV (without pip)"
        "$PY" -m venv --without-pip "$VENV"
        bootstrap_pip
    fi

    # The GPU box routes pip through an internal mirror (see /etc/pip.conf,
    # pypi-mirror.example.com) that answers in ~13s, against pip's 15s default timeout.
    # That is close enough to the edge to fail intermittently, so give it room
    # rather than bypassing the mirror the machine's admins configured.
    # Progress is printed by default. Installing ~100 packages through a mirror
    # that answers in ~13s takes minutes, and a silent terminal is
    # indistinguishable from a hang - which is exactly how this looked the
    # first time. --quiet is opt-in.
    PIP_OPTS=(--timeout 120 --retries 10 --progress-bar on)
    [[ $QUIET -eq 1 ]] && PIP_OPTS+=(--quiet)

    "$VENV/bin/python" -m pip install "${PIP_OPTS[@]}" --upgrade pip
    echo "installing $REQ"
    "$VENV/bin/pip" install "${PIP_OPTS[@]}" -r "$REQ"
    echo "installing $REQ_DEV"
    "$VENV/bin/pip" install "${PIP_OPTS[@]}" -r "$REQ_DEV"
    if [[ "$PROFILE" == "gpu" ]]; then
        echo "installing $REQ_GPU"
        "$VENV/bin/pip" install "${PIP_OPTS[@]}" -r "$REQ_GPU"
    fi
fi

[[ -x "$VENV/bin/python" ]] || { echo "no environment at $VENV" >&2; exit 1; }

# --- verify: report what imports and what does not --------------------------
# Worth doing every time. A missing module in llm_monkeys shows up here as one
# named line rather than as a traceback halfway through a thousand-candidate run.
echo
"$VENV/bin/python" - "$REPO" <<'PYEOF'
import importlib, sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "llm_monkeys"))

print(f"python {sys.version.split()[0]} at {sys.executable}")
import importlib.metadata as meta
for p in ("datasets", "litellm", "google-adk", "google-cloud-aiplatform", "pytest", "pytest-asyncio"):
    try:
        print(f"  {p:<26} {meta.version(p)}")
    except meta.PackageNotFoundError:
        print(f"  {p:<26} MISSING")

modules = [
    "config", "dataset", "model_factory",
    "one_shot.agent", "one_shot.workflow", "one_shot.parser",
    "inference.agent", "inference.parser", "inference.workflow",
    "verifier.agent", "verifier.parser", "verifier.workflow",
    "separate_result", "analyze_results",
]
broken = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        broken.append((name, f"{type(exc).__name__}: {exc}"))

print()
if broken:
    print(f"{len(broken)} of {len(modules)} modules fail to import:")
    for name, err in broken:
        print(f"  {name:<20} {err}")
    sys.exit(1)
print(f"all {len(modules)} modules import cleanly")
PYEOF
