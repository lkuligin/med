#!/usr/bin/env bash
# Build this package's environment.
#
#   ./gpu_serving/setup_env.sh          the serving environment, on the GPU box
#   ./gpu_serving/setup_env.sh --dev    just enough to run the tests, anywhere
#
# Safe to re-run. The venv lives beside this script and is excluded from both
# git and the mirror, so every machine builds its own.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${GPU_SERVING_VENV:-$HERE/.venv}"

# Python 3.12, not 3.13, and this is not a preference. sglang pulls in
# outlines-core, which ships no wheel for 3.13: uv then builds it from source,
# which needs a Rust with edition2024 support, which this box's cargo 1.75 is
# not. On 3.12 the wheel exists and nothing is compiled. The official SGLang
# 0.5.19 image ships 3.12 for the same reason, so it is also the better-tested
# target.
PYTHON_VERSION="${GPU_SERVING_PYTHON_VERSION:-3.12}"

DEV=0
[[ "${1:-}" == "--dev" ]] && DEV=1

if (( DEV )); then
    REQUIREMENTS="$HERE/requirements-dev.txt"
    PROMPT="${GPU_SERVING_ENV_NAME:-serving_dev_env}"
else
    REQUIREMENTS="$HERE/requirements.txt"
    # Shown in the shell prompt once activated. The pipeline's venv calls
    # itself monkeys_env, and on the box both are a tmux window apart, so the
    # prompt is the fastest way to tell which shell can launch a server.
    PROMPT="${GPU_SERVING_ENV_NAME:-serving_env}"
    if ! command -v nvidia-smi >/dev/null; then
        echo "no NVIDIA driver here, so the serving wheels cannot install." >&2
        echo "For a laptop you want:  ./gpu_serving/setup_env.sh --dev" >&2
        exit 1
    fi
fi

echo "building $VENV ($PROMPT) on Python $PYTHON_VERSION from $(basename "$REQUIREMENTS")"

# uv is how the box builds environments, and it can fetch an interpreter the
# system does not have. The dev environment falls back to the standard library
# so a laptop without uv can still run the tests.
if command -v uv >/dev/null; then
    uv venv --python "$PYTHON_VERSION" --prompt "$PROMPT" "$VENV"
    uv pip install --python "$VENV/bin/python" --prerelease=allow -r "$REQUIREMENTS"
elif (( DEV )); then
    echo "uv not found; falling back to the standard library venv module"
    "python$PYTHON_VERSION" -m venv --prompt "$PROMPT" "$VENV" 2>/dev/null \
        || python3 -m venv --prompt "$PROMPT" "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/python" -m pip install --quiet -r "$REQUIREMENTS"
else
    echo "uv not found on PATH; it is what builds the serving environment" >&2
    exit 1
fi

echo
if (( DEV )); then
    "$VENV/bin/python" -m pytest "$HERE/tests" -q
    echo
    echo "done: $PROMPT. The tests above ran without a GPU, which is the point."
else
    "$VENV/bin/python" "$HERE/_verify_install.py"
    echo
    echo "done: $PROMPT. Next: ./gpu_serving/serve.sh list"
fi
