"""Patient EHR, psychological profile, and seed question catalog.

Discovers synthetic patients from either a local directory (``PATIENT_DATA_DIR``,
default ``./data``) or a GCS prefix (``GCS_DATA_PREFIX``). Exposes compact
UI-safe summaries while keeping full raw EHR JSON and profile texts server-side
for prompt construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import random
import re
from typing import Any, TypedDict

DEFAULT_PATIENT_DATA_DIR = "./data"
DEFAULT_PREVIEW_MAX_CHARS = 160
DEFAULT_TEXT_LABEL_MAX_CHARS = 80

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


class ProfileSummary(TypedDict):
    """UI-safe summary of a single pre-generated psychological profile."""

    profile_id: int
    preview: str


class PatientSummary(TypedDict):
    """UI-safe summary of a synthetic patient and its available profiles."""

    ehr_id: str
    display_label: str
    has_seed_questions: bool
    seed_question_count: int
    profiles: list[ProfileSummary]


@dataclass(frozen=True)
class PatientRecord:
    """Server-side patient record holding full EHR JSON, profiles, and seed questions."""

    ehr_id: str
    display_label: str
    ehr_data: dict[str, Any]
    ehr_text: str
    profiles: tuple[str, ...] = field(default_factory=tuple)
    questions: tuple[str, ...] = field(default_factory=tuple)

    def to_summary(self, preview_max_chars: int = DEFAULT_PREVIEW_MAX_CHARS) -> PatientSummary:
        """Returns a UI-safe summary without leaking full EHR JSON or full profile text."""
        return {
            "ehr_id": self.ehr_id,
            "display_label": self.display_label,
            "has_seed_questions": len(self.questions) > 0,
            "seed_question_count": len(self.questions),
            "profiles": [
                {
                    "profile_id": idx,
                    "preview": make_profile_preview(profile, max_chars=preview_max_chars),
                }
                for idx, profile in enumerate(self.profiles)
            ],
        }


def make_profile_preview(
    profile_text: str,
    max_chars: int = DEFAULT_PREVIEW_MAX_CHARS,
) -> str:
    """Creates a clean single-line preview snippet from a psychological profile."""
    collapsed = _WHITESPACE_RE.sub(" ", (profile_text or "").strip())
    if len(collapsed) <= max_chars:
        return collapsed
    cutoff = max(0, max_chars - 3)
    return collapsed[:cutoff].rstrip() + "..."


def _find_fhir_patient_resource(ehr_data: Any) -> dict[str, Any] | None:
    """Finds the FHIR Patient resource in a Bundle or standalone Patient JSON."""
    if not isinstance(ehr_data, dict):
        return None
    resource_type = ehr_data.get("resourceType")
    if resource_type == "Patient":
        return ehr_data
    if resource_type == "Bundle":
        for entry in ehr_data.get("entry") or []:
            if not isinstance(entry, dict):
                continue
            resource = entry.get("resource")
            if isinstance(resource, dict) and resource.get("resourceType") == "Patient":
                return resource
    return None


def _extract_human_name(name_field: Any) -> str | None:
    """Extracts a readable display name from a FHIR Patient.name field."""
    candidates = name_field if isinstance(name_field, list) else [name_field]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return _WHITESPACE_RE.sub(" ", item.strip())
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return _WHITESPACE_RE.sub(" ", text.strip())

        parts: list[str] = []
        for field_name in ("prefix", "given", "family", "suffix"):
            val = item.get(field_name)
            if isinstance(val, list):
                parts.extend(str(p).strip() for p in val if str(p).strip())
            elif isinstance(val, str) and val.strip():
                parts.append(val.strip())
        if parts:
            return _WHITESPACE_RE.sub(" ", " ".join(parts))
    return None


def _extract_narrative_text(text_field: Any) -> str | None:
    """Extracts plain text from a FHIR Patient.text narrative object or string."""
    raw = None
    if isinstance(text_field, str):
        raw = text_field
    elif isinstance(text_field, dict):
        div = text_field.get("div")
        if isinstance(div, str):
            raw = div

    if not raw:
        return None
    stripped = _HTML_TAG_RE.sub(" ", raw)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()
    if not collapsed:
        return None
    if len(collapsed) <= DEFAULT_TEXT_LABEL_MAX_CHARS:
        return collapsed
    return collapsed[: DEFAULT_TEXT_LABEL_MAX_CHARS - 3].rstrip() + "..."


def extract_display_label(ehr_data: Any, fallback_ehr_id: str) -> str:
    """Extracts a display label from FHIR Patient.name / Patient.text, falling back to ehr_id."""
    patient = _find_fhir_patient_resource(ehr_data)
    if patient is not None:
        name_label = _extract_human_name(patient.get("name"))
        if name_label:
            return name_label
        text_label = _extract_narrative_text(patient.get("text"))
        if text_label:
            return text_label
    return fallback_ehr_id


def _parse_json_lenient(raw_text: str) -> Any:
    """Parses JSON, falling back to stripping trailing commas like ``_simulate_chat.py``."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        cleaned = _TRAILING_COMMA_RE.sub(r"\1", raw_text)
        return json.loads(cleaned)


def _extract_profiles_list(data: Any) -> list[str]:
    if isinstance(data, dict) and isinstance(data.get("profiles"), list):
        return [str(p) for p in data["profiles"]]
    if isinstance(data, list):
        return [str(p) for p in data]
    return []


def _extract_questions_list(data: Any) -> list[str]:
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return [str(q) for q in data["questions"]]
    if isinstance(data, list):
        return [str(q) for q in data]
    return []


def _match_questions_for_ehr(
    ehr_id: str,
    questions_map: dict[str, list[str]],
) -> list[str]:
    """Matches a questions list to ``ehr_id`` by exact key or prefix (e.g. ``Hannah`` -> ``Hannah_BRCA2``)."""
    if ehr_id in questions_map:
        return questions_map[ehr_id]
    lower_ehr = ehr_id.lower()
    for q_key, questions in questions_map.items():
        lower_key = q_key.lower()
        if lower_ehr == lower_key or lower_ehr.startswith(f"{lower_key}_") or lower_ehr.startswith(f"{lower_key}-"):
            return questions
    return []


def _classify_json_filename(filename: str) -> tuple[str, str] | None:
    """Classifies a JSON filename into ``(kind, ehr_id)``.

    Returns one of ``("profiles", ehr_id)``, ``("questions", ehr_id)``,
    ``("ehr", ehr_id)``, or ``None`` for non-catalog JSON files (e.g. simulations).
    """
    base = os.path.basename(filename)
    if not base.endswith(".json"):
        return None
    stem = base[:-5]
    if not stem or stem.startswith("simulations_"):
        return None
    if stem.startswith("profiles_"):
        ehr_id = stem.removeprefix("profiles_")
        return ("profiles", ehr_id) if ehr_id else None
    if stem.endswith("_questions"):
        ehr_id = stem.removesuffix("_questions")
        return ("questions", ehr_id) if ehr_id else None
    if stem.startswith("questions_"):
        ehr_id = stem.removeprefix("questions_")
        return ("questions", ehr_id) if ehr_id else None
    return ("ehr", stem)


def _parse_gcs_prefix(
    gcs_data_prefix: str,
    gcs_bucket: str | None = None,
) -> tuple[str, str]:
    """Parses ``gs://bucket/prefix`` or ``(gcs_bucket, prefix)`` into ``(bucket, prefix)``."""
    trimmed = gcs_data_prefix.strip()
    if trimmed.startswith("gs://"):
        without_scheme = trimmed[len("gs://") :]
        parts = without_scheme.split("/", 1)
        bucket = parts[0]
        prefix = parts[1].lstrip("/") if len(parts) > 1 else ""
        if not bucket:
            raise ValueError(f"Invalid GCS URI: {gcs_data_prefix!r}")
        return bucket, prefix

    if not gcs_bucket:
        raise ValueError(
            "GCS_BUCKET must be provided when GCS_DATA_PREFIX is not a gs:// URI."
        )
    return gcs_bucket, trimmed.lstrip("/")


class PatientCatalog:
    """Loads and serves synthetic patient EHRs, profiles, and optional seed questions."""

    def __init__(
        self,
        *,
        data_dir: str | None = None,
        gcs_data_prefix: str | None = None,
        gcs_bucket: str | None = None,
        gcs_client: Any | None = None,
        preview_max_chars: int = DEFAULT_PREVIEW_MAX_CHARS,
    ) -> None:
        self.data_dir = (
            data_dir
            if data_dir is not None
            else os.getenv("PATIENT_DATA_DIR", DEFAULT_PATIENT_DATA_DIR)
        )
        self.gcs_data_prefix = (
            gcs_data_prefix
            if gcs_data_prefix is not None
            else os.getenv("GCS_DATA_PREFIX")
        )
        self.gcs_bucket = (
            gcs_bucket if gcs_bucket is not None else os.getenv("GCS_BUCKET")
        )
        self._gcs_client = gcs_client
        self.preview_max_chars = preview_max_chars
        self._records: dict[str, PatientRecord] = {}
        self.reload()

    def reload(self) -> None:
        """Discovers and loads all patient records from GCS prefix or local directory."""
        if self.gcs_data_prefix:
            raw_files = self._read_files_from_gcs()
        else:
            raw_files = self._read_files_from_local_dir()
        self._records = self._build_records(raw_files)

    def _read_files_from_local_dir(self) -> list[tuple[str, str]]:
        """Reads all JSON files under ``self.data_dir`` as ``(filename, content)`` pairs."""
        if not self.data_dir or not os.path.isdir(self.data_dir):
            return []

        entries: list[tuple[str, str]] = []
        for root, dirs, files in os.walk(self.data_dir):
            dirs.sort()
            for name in sorted(files):
                if not name.endswith(".json"):
                    continue
                full_path = os.path.join(root, name)
                with open(full_path, "r", encoding="utf-8") as f:
                    entries.append((name, f.read()))
        return entries

    def _read_files_from_gcs(self) -> list[tuple[str, str]]:
        """Reads all JSON blobs under ``self.gcs_data_prefix`` as ``(filename, content)``."""
        assert self.gcs_data_prefix is not None
        bucket_name, prefix = _parse_gcs_prefix(self.gcs_data_prefix, self.gcs_bucket)

        client = self._gcs_client
        if client is None:
            from google.cloud import storage

            client = storage.Client()
            self._gcs_client = client

        bucket = client.bucket(bucket_name)
        blobs = client.list_blobs(bucket, prefix=prefix)

        entries: list[tuple[str, str]] = []
        for blob in blobs:
            name = getattr(blob, "name", "")
            if not name.endswith(".json"):
                continue
            entries.append((os.path.basename(name), blob.download_as_text(encoding="utf-8")))
        return entries

    def _build_records(
        self, raw_files: list[tuple[str, str]]
    ) -> dict[str, PatientRecord]:
        ehr_map: dict[str, dict[str, Any]] = {}
        profiles_map: dict[str, list[str]] = {}
        questions_map: dict[str, list[str]] = {}

        for filename, content in raw_files:
            classified = _classify_json_filename(filename)
            if classified is None:
                continue
            kind, ehr_id = classified
            try:
                parsed = _parse_json_lenient(content)
            except (json.JSONDecodeError, ValueError):
                continue

            if kind == "ehr":
                if isinstance(parsed, dict):
                    ehr_map[ehr_id] = parsed
            elif kind == "profiles":
                profiles_map[ehr_id] = _extract_profiles_list(parsed)
            elif kind == "questions":
                questions_map[ehr_id] = _extract_questions_list(parsed)

        records: dict[str, PatientRecord] = {}
        for ehr_id in sorted(ehr_map):
            ehr_data = ehr_map[ehr_id]
            ehr_text = json.dumps(ehr_data, ensure_ascii=False, indent=2)
            display_label = extract_display_label(ehr_data, fallback_ehr_id=ehr_id)
            matched_questions = _match_questions_for_ehr(ehr_id, questions_map)
            records[ehr_id] = PatientRecord(
                ehr_id=ehr_id,
                display_label=display_label,
                ehr_data=ehr_data,
                ehr_text=ehr_text,
                profiles=tuple(profiles_map.get(ehr_id, ())),
                questions=tuple(matched_questions),
            )
        return records

    def list_catalog(self) -> list[PatientSummary]:
        """Returns the UI-safe summary list of all discovered patients and profiles."""
        return [
            record.to_summary(preview_max_chars=self.preview_max_chars)
            for record in self._records.values()
        ]

    def get_patient(self, ehr_id: str) -> PatientRecord:
        """Returns the full server-side ``PatientRecord`` for ``ehr_id``."""
        if ehr_id not in self._records:
            raise KeyError(f"Unknown patient ehr_id: {ehr_id!r}")
        return self._records[ehr_id]

    def get_ehr_text(self, ehr_id: str) -> str:
        """Returns the formatted raw EHR JSON string for prompt construction."""
        return self.get_patient(ehr_id).ehr_text

    def get_profile_text(self, ehr_id: str, profile_id: int) -> str:
        """Returns the full psychological profile text for ``(ehr_id, profile_id)``."""
        record = self.get_patient(ehr_id)
        if profile_id < 0 or profile_id >= len(record.profiles):
            raise IndexError(
                f"profile_id {profile_id} out of range for ehr_id {ehr_id!r} "
                f"(available: 0..{len(record.profiles) - 1})"
            )
        return record.profiles[profile_id]

    def sample_seed_question(
        self,
        ehr_id: str,
        rng: random.Random | None = None,
    ) -> str | None:
        """Returns a random opening question for ``ehr_id``, or ``None`` if none exist."""
        record = self.get_patient(ehr_id)
        if not record.questions:
            return None
        chooser = rng or random
        return chooser.choice(record.questions)
