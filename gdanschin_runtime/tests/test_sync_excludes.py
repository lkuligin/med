"""That the mirror cannot delete what only the GPU box has.

sync.sh runs rsync with --delete, so anything present on the remote and absent
here is removed - unless it is excluded. Excluded paths on the receiver are
protected from --delete; only --delete-excluded would take them, and that flag
is never passed.

The whole protection therefore rests on two things: each remote-owned path
appearing in the exclude list, and that flag staying absent. Both are one edit
away from silently deleting a virtualenv, a results tree, or the log a server
is writing to, with nothing failing until someone looks. Hence these tests.
"""

from pathlib import Path

import pytest

SYNC = Path(__file__).resolve().parents[1] / "sync"
EXCLUDES = SYNC / "rsync-exclude.txt"
SYNC_SH = SYNC / "sync.sh"

# Written on the box, absent here, and expensive or impossible to recreate.
REMOTE_OWNED = {
    "data/": "datasets and downloaded inputs",
    "results/": "experiment output, the point of the runs",
    "exports/": "exported results",
    "cache/": "the Hugging Face cache the box reads datasets from",
    "logs/": "run logs",
    "scratch/": "ad-hoc scripts written on the box",
    "notebooks/": "notebooks filled with outputs there, not here",
    "gpu_serving/run/": "the serving state file and the log a live server writes",
    ".venv/": "environments built against the box's own CUDA",
    ".venv-next/": "the second serving environment, SGLang 0.5.20",
}


@pytest.fixture(scope="module")
def commands() -> str:
    """sync.sh with its comments removed.

    The comments discuss --delete-excluded at length, explaining why it is
    never passed, so a test that greps the whole file finds the very flag it
    is checking for.
    """
    lines = SYNC_SH.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


@pytest.fixture(scope="module")
def patterns() -> set[str]:
    lines = EXCLUDES.read_text().splitlines()
    return {line.strip() for line in lines
            if line.strip() and not line.lstrip().startswith("#")}


@pytest.mark.parametrize("path,why", sorted(REMOTE_OWNED.items()))
def test_a_remote_owned_path_is_excluded(patterns, path, why):
    assert path in patterns, f"{path} is not excluded, and holds {why}"


def test_the_mirror_never_passes_delete_excluded(commands):
    """The one flag that would undo every exclusion above."""
    assert "--delete-excluded" not in commands


def test_the_mirror_still_deletes_what_it_should(commands):
    """Not a mirror otherwise: a module deleted here must not linger there."""
    assert "--delete" in commands


def test_the_exclude_list_is_the_only_one_used(commands):
    """Both transfer and deletion read the same file, so a path cannot be
    protected from one and not the other."""
    assert "--exclude-from=" in commands
    assert "rsync-exclude.txt" in SYNC_SH.read_text()
