"""What this machine can and cannot do with the package.

Most of `gpu_serving` is useful on a laptop: read the catalogue, print the
command a model would be served with, summarise a log pulled back from the box.
Only starting a server needs the machine to actually have the cards, the
driver and the environment.

So rather than let those commands fail somewhere deep - a FileNotFoundError on
nvidia-smi, or a traceback out of uv - the reasons are collected up front and
reported as a list of what is missing. A machine that cannot serve should say
so in one line and exit, not produce a stack trace that reads like a bug.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from gpu_serving.config import Settings, load


@dataclass(frozen=True)
class Capabilities:
    """What is present here, and what that rules out."""

    driver: bool          # nvidia-smi answers
    cards: tuple[int, ...]  # of our configured cards, the ones the driver sees
    environment: bool     # the serving venv is built
    weights_root: bool    # the weights directory exists

    @property
    def can_serve(self) -> bool:
        return bool(self.driver and self.cards and self.environment
                    and self.weights_root)

    def missing(self, settings: Settings) -> list[str]:
        """Why serving is impossible here, in the order worth fixing."""
        problems: list[str] = []
        if not self.driver:
            problems.append(
                "no NVIDIA driver here (nvidia-smi not found) - this machine "
                "has no GPUs to serve on")
        elif not self.cards:
            problems.append(
                f"the driver sees no cards {list(settings.cards)}; configured "
                f"in {settings.__class__.__name__} as GPU_SERVING_CARDS")
        if not self.weights_root:
            problems.append(f"no weights directory at {settings.models_root}")
        if not self.environment:
            problems.append(
                f"serving environment not built at {settings.venv} - run "
                "./gpu_serving/setup_env.sh on the GPU box")
        return problems


def _visible_cards(settings: Settings) -> tuple[int, ...]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ()
    seen = {int(line) for line in result.stdout.split() if line.strip().isdigit()}
    return tuple(c for c in settings.cards if c in seen)


def inspect(settings: Settings | None = None) -> Capabilities:
    """Look at this machine. Cheap enough to call from any command."""
    settings = settings or load()
    driver = shutil.which("nvidia-smi") is not None
    return Capabilities(
        driver=driver,
        cards=_visible_cards(settings) if driver else (),
        environment=settings.python.is_file(),
        weights_root=settings.models_root.is_dir(),
    )


class CannotServe(RuntimeError):
    """This machine cannot serve, and the message says exactly why."""


def require_serving(settings: Settings | None = None) -> Capabilities:
    """Raise with a readable explanation unless this machine can serve."""
    settings = settings or load()
    capabilities = inspect(settings)
    if not capabilities.can_serve:
        problems = capabilities.missing(settings)
        raise CannotServe(
            "cannot serve on this machine:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nWhat does work here: list, command, stats."
        )
    return capabilities


__all__ = ["Capabilities", "CannotServe", "inspect", "require_serving"]
