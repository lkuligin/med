"""On-disk layout for the results of the three workflow steps.

    <results>/<dataset>/single-step/<run>/question_{n}.json
    <results>/<dataset>/facts-pipeline/<run>/question_{n}/iteration_{i}.json
    <results>/<dataset>/facts-pipeline/<run>/question_{n}/<judge>/iteration_{i}.json

One file per record rather than one file per run. A run used to rewrite its
entire output after every question, so a long run rewrote a file of tens of
megabytes hundreds of times, and anything reading it could catch a half-written
copy. Here a record is written once, atomically, and never touched again:
resuming a run costs only the records that are new, extending one to more
candidates adds files rather than rewriting everything, and a reader always
sees whole records.

`<dataset>` comes first because a question id means nothing without it: MedQA
and MedBullets both number their questions from zero, and a run of one stored
beside a run of the other would silently answer the wrong questions.

`<run>` names the model whose answers these are, and `<judge>` the model that
verified them, so several models can be measured on the same questions without
one run's candidates blending into another's. Both default to the model name
and can be set explicitly when the same model is run under different settings.

The step 3 store reads its input through the step 2 store of the run it is
verifying, and writes its verdicts beside those candidates.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterator

DEFAULT_RESULTS_DIR = "results"
DEFAULT_DATASET_DIR = "med_qa"
DEFAULT_ONE_SHOT_FILE = "results_one_shot_gemma4.json"
DEFAULT_CANDIDATES_FILE = "results_step2_gemma4_candidates.json"
DEFAULT_VERIFIED_FILE = "results_step3_verified.json"
SINGLE_STEP = "single-step"
FACTS_PIPELINE = "facts-pipeline"

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"_(\d+)\.json$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def stored_results(payload: Any) -> list[dict[str, Any]]:
    """The result records inside a stored payload, whatever shape it has.

    Every store here writes ``{"summary": ..., "results": [...]}``, but the
    loaders this replaced also accepted a bare ``[...]`` and said so in their
    docstrings. A file from an older run, another tool or a hand edit is still
    a file someone points this at, so the leniency is kept: reading one shape
    only turns step 3 into a TypeError, and every resume path into a silent
    full re-run that costs a whole generation pass.
    """
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
    return []


class SingleFileResults:
    """A whole run in one JSON file: ``{"summary": ..., "results": [...]}``.

    The layout every entry point here has written since the beginning, kept as
    the default so that an environment which has opted into nothing behaves
    exactly as it did. Written to a neighbouring temporary file and renamed, so
    a reader that arrives mid-write sees the old file or the new one.

    An empty path means "do not write", which is what a test that only wants
    the summary passes.
    """

    def __init__(self, filepath: str | Path | None) -> None:
        self.filepath = filepath

    def __str__(self) -> str:
        return str(self.filepath) if self.filepath else "(nowhere)"

    def save(self, payload: dict[str, Any]) -> None:
        """Write the whole payload, atomically."""
        if not self.filepath:
            return
        _write_json(Path(self.filepath), payload)

    def load(self) -> dict[str, Any] | None:
        """The stored payload, or None when nothing is there to read."""
        if not self.filepath:
            return None
        path = Path(self.filepath)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None


class SingleFileVerification(SingleFileResults):
    """Verdicts in one file, reading the candidates it judges from another.

    Step 3 is the only step with an input as well as an output, and the
    workflow reaches it as ``store.candidates`` whichever layout is in use.
    """

    def __init__(self, filepath: str | Path | None,
                 input_filepath: str | Path | None) -> None:
        super().__init__(filepath)
        self.candidates = SingleFileResults(input_filepath)


def build_store(config: Any) -> Any:
    """The store this configuration writes through.

    One construction point, so that a runtime can supply its own layout the
    way it can already supply its own model client. The override is declared
    in MEDQA_RESULTS_STORE as ``"module.path:callable"`` and is handed the
    configuration; unset, results go to one JSON file per run, as they always
    have.

    The returned object needs ``save(payload)``, ``load()`` and a readable
    ``str()`` - and, for step 3, a ``candidates`` store to read from.
    """
    from config import RESULTS_STORE_ENV, resolve_results_store

    factory = resolve_results_store()
    if factory is not None:
        return factory(config)

    # Asking for a run name and getting a single file is how a run ends up
    # somewhere other than where it was meant to go, with nothing to show for
    # it until the results are looked for and not found.
    if getattr(config, "run_name", None):
        logger.warning(
            "--run-name %r names a directory, but results are going to %s: "
            "no layout that keeps one is selected. Set %s to choose one.",
            config.run_name,
            getattr(config, "output_filepath", "") or "(nowhere)",
            RESULTS_STORE_ENV,
        )
    # Duck-typed rather than isinstance: config imports this module, and
    # importing it back for three class names would make that circular.
    if hasattr(config, "input_filepath"):
        return SingleFileVerification(
            getattr(config, "output_filepath", ""), config.input_filepath
        )
    return SingleFileResults(getattr(config, "output_filepath", ""))


def _dataset_holding(results_dir: str | Path, run_name: str, fallback: str) -> str:
    """Which dataset's copy of a run step 3 is being pointed at.

    The verifier reads candidates out of the store rather than loading a
    dataset, so its own dataset_name stays at whatever the default is - and
    verdicts for a MedBullets run would be filed under MedQA, against a run
    that is not there. Looking for the run says which dataset it belongs to
    without anyone having to remember to say so.
    """
    root = Path(results_dir)
    found = (
        sorted(d.name for d in root.iterdir()
               if d.is_dir() and (d / FACTS_PIPELINE / run_name).is_dir())
        if root.is_dir() else []
    )
    if fallback in found:
        return fallback          # asked for a dataset that has it: that one
    if len(found) == 1:
        return found[0]          # only one has it, whatever the config says
    if not found:
        return fallback          # nothing generated yet; take the config at its word
    raise RuntimeError(
        f"{run_name!r} is stored under {', '.join(found)}, and none of those is "
        f"{fallback!r}; name the dataset to verify with --dataset"
    )


def per_record(config: Any) -> Any:
    """A directory per run, told apart by what the configuration carries.

    The entry point for MEDQA_RESULTS_STORE="results_store:per_record". Only
    step 3 names a judge and only step 2 counts candidates, so the step is read
    off the configuration rather than off its class: a configuration built by a
    caller this module does not know still lands in the right layout.
    """
    results_dir = config.results_dir
    run_name = config.resolved_run_name
    dataset = dataset_dir_for(getattr(config, "dataset_name", None))
    sampling = sampling_of(config)
    if hasattr(config, "resolved_judge_name"):
        dataset = _dataset_holding(results_dir, run_name, dataset)
        return VerificationResults(
            results_dir, run_name, config.resolved_judge_name, dataset, sampling
        )
    if hasattr(config, "n_candidates"):
        return CandidateResults(results_dir, run_name, dataset, sampling)
    return OneShotResults(results_dir, run_name, dataset, sampling)


# What makes two runs of the same model produce different answers. Read off
# the configuration rather than passed in, so a caller cannot forget it.
SAMPLING_FIELDS = ("resolved_model_name", "temperature", "max_tokens",
                   "n_attempts", "n_candidates", "concurrency",
                   "dataset_name", "dataset_split", "fact_prompt",
                   "answer_prompt", "judge_prompt")


def sampling_of(config: Any) -> dict[str, Any]:
    """The settings that decide what a run produces, as far as it has them."""
    found = {}
    for field in SAMPLING_FIELDS:
        value = getattr(config, field, None)
        if value is not None:
            found["model" if field == "resolved_model_name" else field] = value
    return found


def dataset_dir_for(dataset_name: str | None) -> str:
    """Derive the directory name for a dataset from its identifier.

    Taken from the canonical identifier rather than from whichever alias was
    typed, so that ``medbullets``, ``medbullets_op5`` and
    ``mkieffer/Medbullets`` all land in one place - and so that adding an alias
    tomorrow cannot quietly move a run somewhere else.
    """
    if not dataset_name or not str(dataset_name).strip():
        return DEFAULT_DATASET_DIR
    tail = str(dataset_name).strip().rstrip("/").split("/")[-1]
    return _UNSAFE.sub("-", tail).strip("-").lower() or DEFAULT_DATASET_DIR


def run_name_for(model_name: str) -> str:
    """Derive a directory name from a model name.

    Model names carry provider prefixes and slashes, neither of which belong in
    a directory name: ``vertex_ai/google/gemma-4-26b-a4b-it-maas`` becomes
    ``gemma-4-26b-a4b-it-maas``.

    Args:
        model_name: Model identifier, with or without provider prefixes.

    Returns:
        A name safe to use as a single directory component.
    """
    tail = str(model_name).rstrip("/").split("/")[-1]
    return _UNSAFE.sub("-", tail).strip("-") or "unnamed"


def split_run_path(path: str | Path) -> tuple[Path, str, str]:
    """Split ``<results>/<dataset>/<section>/<run>`` into its parts.

    Lets a command-line tool take the directory a user can see and tab-complete
    and still build the store that reads it.

    Args:
        path: Directory of one run, as the layout at the top of this module.

    Returns:
        (results directory, run name, dataset directory).
    """
    run_dir = Path(path)
    return run_dir.parent.parent.parent, run_dir.name, run_dir.parent.parent.name


def _write_json(path: Path, payload: Any) -> Path:
    """Write JSON to a path atomically, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a neighbour and renamed, so a reader that arrives mid-write
    # sees either the old file or the new one, never half of either.
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)
    return path


def _numbered(directory: Path, prefix: str) -> Iterator[tuple[int, Path]]:
    """Yield (number, path) for ``<prefix>_<number>.json`` in numeric order.

    Sorting the paths themselves would place iteration_10 before iteration_2
    and silently scramble every question with ten or more of anything.
    """
    if not directory.is_dir():
        return
    found: list[tuple[int, Path]] = []
    for path in directory.glob(f"{prefix}_*.json"):
        match = _NUMBER.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    yield from sorted(found)


class _Results:
    """Shared half of the three stores: a directory, and write-once records."""

    section: str = ""

    def __init__(self, results_dir: str | Path, run_name: str,
                 dataset: str = DEFAULT_DATASET_DIR,
                 sampling: dict[str, Any] | None = None) -> None:
        self.results_dir = Path(results_dir)
        self.run_name = run_name
        self.dataset = dataset or DEFAULT_DATASET_DIR
        self.sampling = sampling or {}
        self._written: set[tuple[str, int]] = set()

    def __str__(self) -> str:
        return str(self.directory)

    @property
    def directory(self) -> Path:
        """The directory holding this run's records."""
        return self.results_dir / self.dataset / self.section / self.run_name

    def _record_path(self, question_id: str, index: int) -> Path:
        raise NotImplementedError

    @staticmethod
    def _read_record(path: Path) -> Any:
        """Parse one record, or None when it is missing or unreadable."""
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _is_new(self, question_id: str, index: int) -> bool:
        """Whether a record still needs writing.

        The set alone would not do: a resumed run starts with an empty one, and
        every save is handed the whole accumulated payload, so the first save
        after resuming carries records that are already on disk.
        """
        key = (str(question_id), index)
        if key in self._written:
            return False
        self._written.add(key)
        return not self._record_path(str(question_id), index).exists()

    # --- summaries ---------------------------------------------------------

    @property
    def summary_path(self) -> Path:
        return self.directory / "summary.json"

    def write_summary(self, summary: Any) -> None:
        """Store a run summary, ignoring the None a mid-run save passes.

        Writing the None through would erase the summary describing everything
        finished so far, which is what a resumed run needs to find.

        The sampling settings are added here because nothing else records
        them. Two runs of one model differ by temperature and token budget
        more than by anything else, and without them written down a directory
        can only be compared with another by trusting its name - which is how
        a comparison between runs that sampled differently gets published as
        a difference between models.
        """
        if summary is None:
            return
        if self.sampling and isinstance(summary, dict):
            summary = {**summary, "sampling": self.sampling}
        _write_json(self.summary_path, summary)

    def read_summary(self) -> Any:
        path = self.summary_path
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class OneShotResults(_Results):
    """Step 1: ``single-step/<run>/question_{n}.json``."""

    section = SINGLE_STEP

    def __init__(self, results_dir: str | Path, run_name: str,
                 dataset: str = DEFAULT_DATASET_DIR,
                 sampling: dict[str, Any] | None = None) -> None:
        super().__init__(results_dir, run_name, dataset, sampling)
        self._stored: dict[str, Any] = {}

    def question_path(self, question_id: int | str) -> Path:
        return self.directory / f"question_{question_id}.json"

    def _record_path(self, question_id: str, index: int) -> Path:
        return self.question_path(question_id)

    def save(self, payload: dict[str, Any]) -> None:
        """Write one file per question result, plus the run summary.

        A record is rewritten only when it differs from the one on disk. Unlike
        a candidate, which is generated once and then only added to, a question
        that failed is re-processed on the next run and its record has to be
        replaced - and a record that would not parse has to be replaced too.
        """
        for item in payload.get("results") or []:
            question_id = str(item.get("question_id"))
            if self._stored.get(question_id) != item:
                _write_json(self.question_path(question_id), item)
                self._stored[question_id] = item
        self.write_summary(payload.get("summary"))

    def read(self, question_id: int | str | None = None):
        """One question's record, or every record keyed by question id.

        A record that does not parse is left out rather than raised over: it
        costs that one question, which the next run re-processes, instead of
        the whole run.
        """
        if question_id is not None:
            record = self._read_record(self.question_path(question_id))
            if record is not None:
                self._stored[str(question_id)] = record
            return record
        records = {}
        for number, path in _numbered(self.directory, "question"):
            record = self._read_record(path)
            if record is None:
                continue
            # Key by the id the record carries, not by the number parsed out of
            # the file name: int("001") is 1, and that turned medbullets
            # question 001 into "1" here while every other store, the
            # difficult list and the dataset itself all say "001".
            key = str(record.get("question_id", number))
            records[key] = record
            self._stored[key] = record
        return records

    def load(self) -> dict[str, Any] | None:
        """The whole run in one payload, or None when nothing is stored."""
        records = self.read()
        if not records:
            return None
        return {"summary": self.read_summary(), "results": list(records.values())}


class CandidateResults(_Results):
    """Step 2: ``facts-pipeline/<run>/question_{n}/iteration_{i}.json``."""

    section = FACTS_PIPELINE

    def question_dir(self, question_id: int | str) -> Path:
        return self.directory / f"question_{question_id}"

    def candidate_path(self, question_id: int | str, iteration: int) -> Path:
        return self.question_dir(question_id) / f"iteration_{iteration}.json"

    def _record_path(self, question_id: str, index: int) -> Path:
        return self.candidate_path(question_id, index)

    def questions(self) -> list[str]:
        """Question ids this run has candidates for, in numeric order."""
        if not self.directory.is_dir():
            return []
        ids = [
            path.name.removeprefix("question_")
            for path in self.directory.iterdir()
            if path.is_dir()
        ]
        return sorted(ids, key=lambda name: int(name) if name.isdigit() else 0)

    def read_candidates(self, question_id: int | str) -> list[dict[str, Any]]:
        """Every readable candidate for one question, in iteration order."""
        records = (
            self._read_record(path)
            for _, path in _numbered(self.question_dir(question_id), "iteration")
        )
        return [record for record in records if record is not None]

    def candidate_count(self, question_id: int | str) -> int:
        return sum(1 for _ in _numbered(self.question_dir(question_id), "iteration"))

    def save(self, payload: dict[str, Any]) -> None:
        """Write one file per candidate, plus the run summary.

        Each candidate file repeats its question, options and ground truth, so
        that a file can be read without the ones around it.
        """
        for question in payload.get("results") or []:
            question_id = str(question.get("question_id"))
            # meta_info and ground_truth_answer come from the dataset rather
            # than from the run, and nothing here can recompute them. They are
            # kept so that a record read on its own, or assembled back into the
            # single-file layout, is as complete as what the workflow held.
            context = {
                "question_id": question_id,
                "meta_info": question.get("meta_info"),
                "question": question.get("question"),
                "options": question.get("options"),
                "ground_truth": question.get("ground_truth"),
                "ground_truth_answer": question.get("ground_truth_answer"),
            }
            for position, candidate in enumerate(question.get("candidates") or []):
                index = candidate.get("candidate_index", position)
                if self._is_new(question_id, index):
                    _write_json(
                        self.candidate_path(question_id, index),
                        {**context, "candidate": candidate},
                    )
        self.write_summary(payload.get("summary"))

    def load(self) -> dict[str, Any] | None:
        """Every question with its candidates, or None when nothing is stored."""
        results = []
        for question_id in self.questions():
            candidates = self.read_candidates(question_id)
            if not candidates:
                continue
            head = candidates[0]
            results.append(
                {
                    "question_id": question_id,
                    "meta_info": head.get("meta_info"),
                    "question": head.get("question"),
                    "options": head.get("options"),
                    "ground_truth": head.get("ground_truth"),
                    "ground_truth_answer": head.get("ground_truth_answer"),
                    "candidates": [record["candidate"] for record in candidates],
                }
            )
        if not results:
            return None
        return {"summary": self.read_summary(), "results": results}


class VerificationResults(_Results):
    """Step 3: ``facts-pipeline/<run>/question_{n}/<judge>/iteration_{i}.json``.

    The verdicts sit beside the candidates they judge, so a question directory
    holds one run of candidates and a subdirectory per judge that has seen them.
    """

    section = FACTS_PIPELINE

    def __init__(self, results_dir: str | Path, run_name: str, judge_name: str,
                 dataset: str = DEFAULT_DATASET_DIR,
                 sampling: dict[str, Any] | None = None) -> None:
        super().__init__(results_dir, run_name, dataset, sampling)
        self.judge_name = judge_name
        self.candidates = CandidateResults(results_dir, run_name, dataset)
        self._outcomes: dict[str, Any] = {}

    def __str__(self) -> str:
        return f"{self.directory} ({self.judge_name})"

    @property
    def summary_path(self) -> Path:
        return self.directory / f"summary.{self.judge_name}.json"

    def judge_dir(self, question_id: int | str) -> Path:
        return self.candidates.question_dir(question_id) / self.judge_name

    def verdict_path(self, question_id: int | str, iteration: int) -> Path:
        return self.judge_dir(question_id) / f"iteration_{iteration}.json"

    def outcome_path(self, question_id: int | str) -> Path:
        """Where the question-level outcome of the judging goes."""
        return self.judge_dir(question_id) / "result.json"

    def _record_path(self, question_id: str, index: int) -> Path:
        return self.verdict_path(question_id, index)

    def read_verdicts(self, question_id: int | str) -> list[dict[str, Any]]:
        records = (
            self._read_record(path)
            for _, path in _numbered(self.judge_dir(question_id), "iteration")
        )
        return [record for record in records if record is not None]

    def judges(self) -> list[str]:
        """Judges that have verdicts somewhere in this run."""
        found: set[str] = set()
        for question_id in self.candidates.questions():
            for path in self.candidates.question_dir(question_id).iterdir():
                if path.is_dir():
                    found.add(path.name)
        return sorted(found)

    def save(self, payload: dict[str, Any]) -> None:
        """Write one file per judged candidate, the question-level outcome of
        each, and the run summary."""
        for question in payload.get("results") or []:
            question_id = str(question.get("question_id"))
            context = {
                "question_id": question_id,
                "ground_truth": question.get("ground_truth"),
                "judge": self.judge_name,
            }
            verifications = question.get("candidate_verifications") or []
            for position, verification in enumerate(verifications):
                index = verification.get("candidate_index", position)
                if self._is_new(question_id, index):
                    _write_json(
                        self.verdict_path(question_id, index),
                        {**context, "verification": verification},
                    )
            # Which candidate was accepted is a fact about the question, not
            # about any one verdict, and it changes when a question that found
            # nothing is later given more candidates - so it is written when it
            # differs, rather than once.
            outcome = {k: v for k, v in question.items()
                       if k != "candidate_verifications"}
            if self._outcomes.get(question_id) != outcome:
                _write_json(self.outcome_path(question_id), outcome)
                self._outcomes[question_id] = outcome
        self.write_summary(payload.get("summary"))

    def load(self) -> dict[str, Any] | None:
        """Every judged question, or None when this judge has seen none."""
        results = []
        for question_id in self.candidates.questions():
            verdicts = self.read_verdicts(question_id)
            if not verdicts:
                continue
            verifications = [r["verification"] for r in verdicts]
            outcome = self._read_record(self.outcome_path(question_id))
            if outcome is None:
                outcome = self._derive_outcome(question_id, verdicts, verifications)
            else:
                self._outcomes[question_id] = outcome
            results.append({**outcome, "candidate_verifications": verifications})
        if not results:
            return None
        return {"summary": self.read_summary(), "results": results}

    @staticmethod
    def _derive_outcome(question_id: str, verdicts: list[dict],
                        verifications: list[dict]) -> dict[str, Any]:
        """Reconstruct what the verdicts imply, for runs stored without an outcome.

        Enough of it to decide whether a question is finished and which
        candidate answered it; the rest (the question text, token totals) is
        only in the outcome file.
        """
        passed = next((v for v in verifications if v.get("all_facts_correct")), None)
        return {
            "question_id": question_id,
            "ground_truth": verdicts[0].get("ground_truth"),
            "candidates_evaluated": len(verifications),
            "found_valid_candidate": passed is not None,
            "selected_candidate_index": passed.get("candidate_index") if passed else None,
            "attempt_number": passed.get("attempt_number") if passed else None,
            "predicted_option": passed.get("predicted_option") if passed else None,
            "is_correct": passed.get("is_correct") if passed else None,
        }


__all__ = [
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_DATASET_DIR",
    "dataset_dir_for",
    "DEFAULT_ONE_SHOT_FILE",
    "DEFAULT_CANDIDATES_FILE",
    "DEFAULT_VERIFIED_FILE",
    "SingleFileResults",
    "SingleFileVerification",
    "build_store",
    "per_record",
    "split_run_path",
    "SINGLE_STEP",
    "FACTS_PIPELINE",
    "CandidateResults",
    "OneShotResults",
    "VerificationResults",
    "run_name_for",
]
