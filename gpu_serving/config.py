"""Where this box keeps its cards, weights and virtualenv.

Everything here is site-specific: the cards we are allowed to use, the path the
weights live under, which Python has SGLang installed. Keeping it in one file
read from configs/serving.conf is what lets this package move to another box -
or another repository - without edits to the code.

Any GPU_SERVING_* environment variable overrides the file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "serving.conf"

DEFAULTS: dict[str, str] = {
    # Cards 0-3 on this box belong to other people. Serving anything on them
    # would collide with someone else's run, so the allocation is configuration
    # rather than a flag anyone can pass by accident.
    "GPU_SERVING_CARDS": "4,5,6,7",
    "GPU_SERVING_MODELS_ROOT": "/mnt/data/models",
    # The package carries its own environment, so that moving this directory
    # moves everything it needs. ".venv" at any depth is already excluded from
    # both git and the mirror, so the box builds its own against its own CUDA.
    "GPU_SERVING_VENV": ".venv",
    "GPU_SERVING_HOST": "127.0.0.1",
    "GPU_SERVING_PORT": "8000",
    # State and server logs. The sync runs with --delete, so this path is
    # listed in rsync-exclude.txt as remote-owned; without that entry it would
    # be removed under a running server on the next mirror.
    "GPU_SERVING_RUN_DIR": "run",
}

_LINE = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)\s*=\s*"?(.*?)"?\s*$')


def _from_file() -> dict[str, str]:
    values: dict[str, str] = {}
    if CONFIG.is_file():
        for line in CONFIG.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = _LINE.match(line)
            if m:
                values[m.group(1)] = m.group(2)
    return values


@dataclass(frozen=True)
class Settings:
    cards: tuple[int, ...]
    models_root: Path
    venv: Path
    host: str
    port: int
    run_dir: Path

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _path(value: str) -> Path:
    """Resolve a setting that may be relative to this package."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else HERE / path


def load() -> Settings:
    """Read settings: environment first, then the config file, then defaults."""
    values = {**DEFAULTS, **_from_file()}
    values.update({k: v for k, v in os.environ.items() if k in DEFAULTS})

    cards = tuple(int(c) for c in values["GPU_SERVING_CARDS"].split(",") if c.strip())
    if not cards:
        raise ValueError("GPU_SERVING_CARDS is empty; no cards to serve on")

    return Settings(
        cards=cards,
        models_root=Path(values["GPU_SERVING_MODELS_ROOT"]),
        venv=_path(values["GPU_SERVING_VENV"]),
        host=values["GPU_SERVING_HOST"],
        port=int(values["GPU_SERVING_PORT"]),
        run_dir=_path(values["GPU_SERVING_RUN_DIR"]),
    )


__all__ = ["Settings", "load", "CONFIG", "DEFAULTS"]
