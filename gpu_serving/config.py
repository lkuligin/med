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
    # Where weights are looked for, in order, as flat <root>/<org>/<name>
    # directories. Colon-separated, like PATH: the shared directory first,
    # then anywhere we keep our own. /mnt/data/models belongs to someone else
    # and is read-only for us, which is why there has to be a second place.
    "GPU_SERVING_MODELS_ROOTS": "/mnt/data/models",
    # A Hugging Face cache, used for models that are not in any flat root:
    # SGLang is then given the repo id and resolves it here. This is also
    # where downloads go, so it has to be writable. /mnt/data/model (singular)
    # is the box's shared cache - world-writable, on the big array, and
    # already holding what the sglang-worker services use.
    "GPU_SERVING_HF_HOME": "/mnt/data/model",
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
    models_roots: tuple[Path, ...]
    hf_home: Path
    venv: Path
    host: str
    port: int
    run_dir: Path

    @property
    def models_root(self) -> Path:
        """The first flat root. Kept for callers that only need somewhere."""
        return self.models_roots[0]

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

    roots = tuple(Path(r).expanduser()
                  for r in values["GPU_SERVING_MODELS_ROOTS"].split(":") if r.strip())
    if not roots:
        raise ValueError("GPU_SERVING_MODELS_ROOTS is empty; nowhere to find weights")

    return Settings(
        cards=cards,
        models_roots=roots,
        hf_home=Path(values["GPU_SERVING_HF_HOME"]).expanduser(),
        venv=_path(values["GPU_SERVING_VENV"]),
        host=values["GPU_SERVING_HOST"],
        port=int(values["GPU_SERVING_PORT"]),
        run_dir=_path(values["GPU_SERVING_RUN_DIR"]),
    )


__all__ = ["Settings", "load", "CONFIG", "DEFAULTS"]
