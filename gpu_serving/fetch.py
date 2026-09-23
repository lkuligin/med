"""Download a model's weights into the cache this box serves from.

The shared directory of flat weight folders belongs to another user and is
read-only for us, so nothing new can be added there. Downloads go instead to
the Hugging Face cache named by GPU_SERVING_HF_HOME - on this box a shared,
world-writable directory on the large array, already holding what the
sglang-worker services use. Putting ours beside them means the next person
does not download them again.

Kept out of server.py because it is the one operation here that reaches the
network, takes tens of gigabytes, and must never happen implicitly as part of
starting a server.
"""

from __future__ import annotations

import os
import subprocess

from gpu_serving.catalog import SERVABLE, ServedModel
from gpu_serving.config import Settings, load

# The download needs the network, which the serving environment is otherwise
# told not to use. Set here rather than globally so a serve never inherits it.
ONLINE = {
    "HF_HUB_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_ETAG_TIMEOUT": "20",
    "HF_HUB_DOWNLOAD_TIMEOUT": "60",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

# Duplicate weights for other runtimes, which double the download for nothing.
EXCLUDE = ("original/*", "metal/*", "consolidated*")


def fetch_argv(model: ServedModel, settings: Settings) -> list[str]:
    """The download command. Pure, so a test can read it without a network."""
    argv = [str(settings.venv / "bin" / "hf"), "download", model.weights]
    for pattern in EXCLUDE:
        argv += ["--exclude", pattern]
    return argv


def fetch_env(settings: Settings) -> dict[str, str]:
    return {**os.environ, "HF_HOME": str(settings.hf_home), **ONLINE}


def fetch(name: str, settings: Settings | None = None) -> int:
    """Download `name` into the cache. Streams progress; returns the exit code."""
    settings = settings or load()
    if name not in SERVABLE:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(SERVABLE)}")
    model = SERVABLE[name]

    if not os.access(settings.hf_home, os.W_OK):
        raise PermissionError(
            f"{settings.hf_home} is not writable by us. Point "
            f"GPU_SERVING_HF_HOME somewhere that is."
        )

    print(f"fetching {model.weights} (~{model.weights_gb:.0f} GB) "
          f"into {settings.hf_home}")
    return subprocess.run(fetch_argv(model, settings),
                          env=fetch_env(settings)).returncode


__all__ = ["fetch", "fetch_argv", "fetch_env", "EXCLUDE"]
