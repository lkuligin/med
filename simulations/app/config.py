"""Environment-based configuration for the interactive patient simulation app."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

from app.catalog import DEFAULT_PATIENT_DATA_DIR
from app.patient_engine import (
    DEFAULT_LANGUAGE,
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    DEFAULT_REPLICA_THRESHOLD,
)
from app.storage import DEFAULT_GCS_CHAT_PREFIX, DEFAULT_LOCAL_STORAGE_DIR

DEFAULT_MAX_TURNS = 50


def _parse_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    gcp_project: str | None = None
    gcp_location: str = DEFAULT_LOCATION
    model_name: str = DEFAULT_MODEL
    default_language: str = DEFAULT_LANGUAGE
    replica_threshold: int = DEFAULT_REPLICA_THRESHOLD
    max_turns: int = DEFAULT_MAX_TURNS
    patient_data_dir: str = DEFAULT_PATIENT_DATA_DIR
    gcs_data_prefix: str = ""
    gcs_bucket: str = ""
    gcs_chat_prefix: str = DEFAULT_GCS_CHAT_PREFIX
    local_storage_dir: str = DEFAULT_LOCAL_STORAGE_DIR

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        project = (
            source.get("GCP_PROJECT")
            or source.get("GOOGLE_CLOUD_PROJECT")
            or ""
        ).strip() or None
        return cls(
            gcp_project=project,
            gcp_location=source.get("GCP_LOCATION", DEFAULT_LOCATION).strip()
            or DEFAULT_LOCATION,
            model_name=source.get("MODEL_NAME", DEFAULT_MODEL).strip()
            or DEFAULT_MODEL,
            default_language=source.get("DEFAULT_LANGUAGE", DEFAULT_LANGUAGE).strip()
            or DEFAULT_LANGUAGE,
            replica_threshold=_parse_int(
                source.get("REPLICA_THRESHOLD"), DEFAULT_REPLICA_THRESHOLD
            ),
            max_turns=_parse_int(source.get("MAX_TURNS"), DEFAULT_MAX_TURNS),
            patient_data_dir=source.get(
                "PATIENT_DATA_DIR", DEFAULT_PATIENT_DATA_DIR
            ).strip()
            or DEFAULT_PATIENT_DATA_DIR,
            gcs_data_prefix=source.get("GCS_DATA_PREFIX", "").strip(),
            gcs_bucket=source.get("GCS_BUCKET", "").strip(),
            gcs_chat_prefix=source.get(
                "GCS_CHAT_PREFIX", DEFAULT_GCS_CHAT_PREFIX
            ).strip()
            or DEFAULT_GCS_CHAT_PREFIX,
            local_storage_dir=source.get(
                "LOCAL_STORAGE_DIR", DEFAULT_LOCAL_STORAGE_DIR
            ).strip()
            or DEFAULT_LOCAL_STORAGE_DIR,
        )
