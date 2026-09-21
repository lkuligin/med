#!/usr/bin/env bash
# Serve one model on this box. See gpu_serving/cli.py for the actions.
#
# Deliberately run with the system python, not the SGLang venv: this package is
# standard library only, and the venv is what it LAUNCHES, not what it needs.
# That keeps "serve.sh status" working even when the venv is half-installed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec "${GPU_SERVING_PYTHON:-python3}" -m gpu_serving.cli "$@"
