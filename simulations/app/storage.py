"""Per-turn session JSON persistence for GCS with local-directory fallback.

Each chat session is stored as a single JSON object at:
- GCS: ``gs://<GCS_BUCKET>/<GCS_CHAT_PREFIX>/<session_id>.json``
- Local dev fallback (when ``GCS_BUCKET`` is unset):
  ``<LOCAL_STORAGE_DIR>/<session_id>.json``

Writes are atomic on both backends (GCS object overwrite / local temp-file
``os.replace``) so overwriting ``<session_id>.json`` on every turn guarantees
no partial or corrupted files and enables stateless reload across Cloud Run
instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timezone
import json
import os
import re
import tempfile
from typing import Any
import uuid

DEFAULT_GCS_CHAT_PREFIX = "chats"
DEFAULT_LOCAL_STORAGE_DIR = "./local_chats"

SESSION_STATUS_ACTIVE = "active"
SESSION_STATUS_ENDED_BY_DOCTOR = "ended_by_doctor"

_SAFE_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class SessionNotFoundError(KeyError):
    """Raised when a requested chat session JSON does not exist in storage."""


def utc_now_iso() -> str:
    """Returns the current UTC timestamp formatted as ISO-8601 with trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_session_id(session_id: str) -> str:
    """Validates that ``session_id`` is a safe single path segment."""
    cleaned = (session_id or "").strip()
    if (
        not cleaned
        or cleaned in (".", "..")
        or not _SAFE_SESSION_ID_RE.match(cleaned)
    ):
        raise ValueError(f"Invalid session_id: {session_id!r}")
    return cleaned


def create_session_record(
    *,
    doctor_email: str,
    doctor_user_id: str,
    ehr_id: str,
    profile_id: int,
    profile_text: str,
    model: str,
    language: str = "de",
    replica_threshold: int = 300,
    max_turns: int = 50,
    feed_question: str | None = None,
    session_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Builds a new chat session JSON document matching the persistence schema."""
    resolved_id = validate_session_id(session_id or str(uuid.uuid4()))
    timestamp = created_at or utc_now_iso()
    return {
        "session_id": resolved_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": SESSION_STATUS_ACTIVE,
        "doctor": {
            "email": doctor_email,
            "user_id": doctor_user_id,
        },
        "patient": {
            "ehr_id": ehr_id,
            "profile_id": profile_id,
            "profile_text": profile_text,
        },
        "config": {
            "model": model,
            "language": language,
            "replica_threshold": replica_threshold,
            "max_turns": max_turns,
        },
        "metadata": {
            "feed_question": feed_question,
        },
        "turns": [],
    }


def append_turn(
    session: dict[str, Any],
    *,
    role: str,
    text: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Appends a new doctor or patient turn to ``session`` and updates ``updated_at``."""
    turns = session.setdefault("turns", [])
    turn_ts = timestamp or utc_now_iso()
    turn: dict[str, Any] = {
        "turn_index": len(turns),
        "role": role,
        "text": text,
        "timestamp": turn_ts,
        "source": source,
        "metadata": dict(metadata) if metadata else {},
    }
    turns.append(turn)
    session["updated_at"] = turn_ts
    return turn


def mark_session_ended(
    session: dict[str, Any],
    *,
    status: str = SESSION_STATUS_ENDED_BY_DOCTOR,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Marks ``session`` as ended and updates ``updated_at``."""
    session["status"] = status
    session["updated_at"] = timestamp or utc_now_iso()
    return session


def _serialize_session(session: dict[str, Any]) -> str:
    return json.dumps(session, ensure_ascii=False, indent=2)


def _extract_feed_question(session: dict[str, Any]) -> str | None:
    """Extracts the session's feed question from top-level metadata or opening seed turn."""
    meta = session.get("metadata")
    if isinstance(meta, dict):
        fq = meta.get("feed_question")
        if isinstance(fq, str) and fq.strip():
            return fq.strip()
    turns = session.get("turns")
    if isinstance(turns, list) and turns:
        first = turns[0]
        if isinstance(first, dict) and first.get("source") == "seed_question":
            text = first.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _build_gcs_blob_metadata(session: dict[str, Any]) -> dict[str, str]:
    """Builds string-only custom metadata for the GCS session JSON object."""
    metadata: dict[str, str] = {}
    patient = session.get("patient")
    if isinstance(patient, dict):
        ehr_id = patient.get("ehr_id")
        if ehr_id is not None and str(ehr_id).strip():
            metadata["ehr_id"] = str(ehr_id).strip()
        profile_id = patient.get("profile_id")
        if profile_id is not None:
            metadata["profile_id"] = str(profile_id)

    feed_question = _extract_feed_question(session)
    if feed_question:
        metadata["feed_question"] = " ".join(feed_question.splitlines()).strip()

    return metadata


class SessionStore(ABC):
    """Abstract interface for per-turn chat session JSON persistence."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for this backend (``"gcs"`` or ``"local"``)."""

    @abstractmethod
    def describe_location(self, session_id: str) -> str:
        """Returns a human-readable URI or path for ``session_id``."""

    @abstractmethod
    def save(self, session: dict[str, Any]) -> str:
        """Atomically writes the full session JSON and returns its storage location."""

    @abstractmethod
    def load(self, session_id: str) -> dict[str, Any]:
        """Loads and parses the session JSON, raising ``SessionNotFoundError`` if absent."""

    async def asave(self, session: dict[str, Any]) -> str:
        """Async wrapper around ``save`` that avoids blocking the event loop."""
        return await asyncio.to_thread(self.save, session)

    async def aload(self, session_id: str) -> dict[str, Any]:
        """Async wrapper around ``load`` that avoids blocking the event loop."""
        return await asyncio.to_thread(self.load, session_id)


class GCSSessionStore(SessionStore):
    """Stores each chat session JSON at ``gs://<bucket>/<prefix>/<session_id>.json``."""

    def __init__(
        self,
        bucket_name: str,
        *,
        prefix: str = DEFAULT_GCS_CHAT_PREFIX,
        client: Any | None = None,
    ) -> None:
        cleaned_bucket = (bucket_name or "").strip()
        if not cleaned_bucket:
            raise ValueError("bucket_name must be a non-empty GCS bucket name.")
        self.bucket_name = cleaned_bucket
        self.prefix = (prefix or "").strip("/")
        self._client = client

    @property
    def backend_name(self) -> str:
        return "gcs"

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        return self._client

    def blob_path(self, session_id: str) -> str:
        safe_id = validate_session_id(session_id)
        if self.prefix:
            return f"{self.prefix}/{safe_id}.json"
        return f"{safe_id}.json"

    def describe_location(self, session_id: str) -> str:
        return f"gs://{self.bucket_name}/{self.blob_path(session_id)}"

    def save(self, session: dict[str, Any]) -> str:
        session_id = validate_session_id(str(session.get("session_id", "")))
        blob_name = self.blob_path(session_id)
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        blob_metadata = _build_gcs_blob_metadata(session)
        if blob_metadata:
            blob.metadata = blob_metadata
        blob.upload_from_string(
            _serialize_session(session),
            content_type="application/json",
        )
        return self.describe_location(session_id)

    def load(self, session_id: str) -> dict[str, Any]:
        safe_id = validate_session_id(session_id)
        blob_name = self.blob_path(safe_id)
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise SessionNotFoundError(f"Session not found: {safe_id!r}")
        raw_text = blob.download_as_text(encoding="utf-8")
        return json.loads(raw_text)


class LocalSessionStore(SessionStore):
    """Local-directory session store for development when ``GCS_BUCKET`` is unset."""

    def __init__(self, storage_dir: str = DEFAULT_LOCAL_STORAGE_DIR) -> None:
        cleaned_dir = (storage_dir or "").strip() or DEFAULT_LOCAL_STORAGE_DIR
        self.storage_dir = cleaned_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    @property
    def backend_name(self) -> str:
        return "local"

    def file_path(self, session_id: str) -> str:
        safe_id = validate_session_id(session_id)
        return os.path.join(self.storage_dir, f"{safe_id}.json")

    def describe_location(self, session_id: str) -> str:
        return self.file_path(session_id)

    def save(self, session: dict[str, Any]) -> str:
        session_id = validate_session_id(str(session.get("session_id", "")))
        target_path = self.file_path(session_id)
        os.makedirs(self.storage_dir, exist_ok=True)
        payload = _serialize_session(session)

        # Atomic write via temporary file in the same directory + os.replace.
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{session_id}.",
            suffix=".tmp",
            dir=self.storage_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(payload)
            os.replace(tmp_path, target_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return target_path

    def load(self, session_id: str) -> dict[str, Any]:
        safe_id = validate_session_id(session_id)
        target_path = self.file_path(safe_id)
        if not os.path.isfile(target_path):
            raise SessionNotFoundError(f"Session not found: {safe_id!r}")
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)


def create_session_store(
    *,
    gcs_bucket: str | None = None,
    gcs_chat_prefix: str | None = None,
    local_storage_dir: str | None = None,
    gcs_client: Any | None = None,
) -> SessionStore:
    """Creates a ``GCSSessionStore`` when ``GCS_BUCKET`` is set, else ``LocalSessionStore``."""
    resolved_bucket = (
        os.getenv("GCS_BUCKET", "").strip()
        if gcs_bucket is None
        else gcs_bucket.strip()
    )
    if resolved_bucket:
        resolved_prefix = (
            os.getenv("GCS_CHAT_PREFIX", DEFAULT_GCS_CHAT_PREFIX)
            if gcs_chat_prefix is None
            else gcs_chat_prefix
        )
        return GCSSessionStore(
            resolved_bucket,
            prefix=resolved_prefix,
            client=gcs_client,
        )

    resolved_local_dir = (
        os.getenv("LOCAL_STORAGE_DIR", DEFAULT_LOCAL_STORAGE_DIR)
        if local_storage_dir is None
        else local_storage_dir
    )
    return LocalSessionStore(resolved_local_dir)
