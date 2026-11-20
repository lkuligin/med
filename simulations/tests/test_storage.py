"""Unit tests for app.storage."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

_SIMULATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIMULATIONS_DIR not in sys.path:
    sys.path.insert(0, _SIMULATIONS_DIR)

from app import storage  # noqa: E402


class _InMemoryBlob:
    def __init__(self, store: dict[str, str], name: str) -> None:
        self._store = store
        self.name = name
        self.last_content_type: str | None = None
        self.metadata: dict[str, str] | None = None

    def exists(self) -> bool:
        return self.name in self._store

    def upload_from_string(self, data: str, content_type: str = "") -> None:
        self._store[self.name] = data
        self.last_content_type = content_type

    def download_as_text(self, encoding: str = "utf-8") -> str:
        return self._store[self.name]


class _InMemoryBucket:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}
        self.blobs: dict[str, _InMemoryBlob] = {}

    def blob(self, name: str) -> _InMemoryBlob:
        if name not in self.blobs:
            self.blobs[name] = _InMemoryBlob(self.objects, name)
        return self.blobs[name]


class _InMemoryGCSClient:
    def __init__(self) -> None:
        self.buckets: dict[str, _InMemoryBucket] = {}

    def bucket(self, name: str) -> _InMemoryBucket:
        if name not in self.buckets:
            self.buckets[name] = _InMemoryBucket()
        return self.buckets[name]


class TestSessionSchemaHelpers(unittest.TestCase):
    def test_session_lifecycle_matches_schema(self) -> None:
        session = storage.create_session_record(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            created_at="2026-11-20T11:00:00Z",
            doctor_email="doctor@example.com",
            doctor_user_id="109876543210",
            ehr_id="sarah-levi-brca1",
            profile_id=4,
            profile_text="Besorgte Patientin...",
            model="gemini-3.8-flash",
            language="de",
            replica_threshold=300,
            max_turns=50,
        )

        self.assertEqual(session["status"], "active")
        self.assertEqual(session["doctor"], {"email": "doctor@example.com", "user_id": "109876543210"})
        self.assertEqual(
            session["patient"],
            {
                "ehr_id": "sarah-levi-brca1",
                "profile_id": 4,
                "profile_text": "Besorgte Patientin...",
            },
        )
        self.assertEqual(
            session["config"],
            {
                "model": "gemini-3.8-flash",
                "language": "de",
                "replica_threshold": 300,
                "max_turns": 50,
            },
        )

        t0 = storage.append_turn(
            session,
            role="patient",
            text="Hallo, ich bin ständig erschöpft.",
            source="seed_question",
            timestamp="2026-11-20T11:00:01Z",
        )
        t1 = storage.append_turn(
            session,
            role="doctor",
            text="Seit wann ist die Müdigkeit besonders stark?",
            source="human",
            timestamp="2026-11-20T11:01:05Z",
        )
        t2 = storage.append_turn(
            session,
            role="patient",
            text="Eigentlich fast durchgehend seit etwa einem Monat.",
            source="llm",
            metadata={"char_count": 50, "regenerated": False},
            timestamp="2026-11-20T11:01:09Z",
        )
        storage.mark_session_ended(session, timestamp="2026-11-20T11:03:12Z")

        self.assertEqual([t0["turn_index"], t1["turn_index"], t2["turn_index"]], [0, 1, 2])
        self.assertEqual(t2["metadata"], {"char_count": 50, "regenerated": False})
        self.assertEqual(session["status"], "ended_by_doctor")
        self.assertEqual(session["updated_at"], "2026-11-20T11:03:12Z")

    def test_validate_session_id_rejects_traversal(self) -> None:
        for bad in ("", "   ", "..", ".", "../evil", "a/b", "a\\b"):
            with self.assertRaises(ValueError):
                storage.validate_session_id(bad)


class TestLocalSessionStore(unittest.IsolatedAsyncioTestCase):
    async def test_per_turn_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = storage.LocalSessionStore(tmpdir)
            self.assertEqual(store.backend_name, "local")

            session = storage.create_session_record(
                session_id="sess-local-1",
                doctor_email="dev-doctor@local",
                doctor_user_id="local-dev-user",
                ehr_id="sarah-levi-brca1",
                profile_id=0,
                profile_text="Profil 0",
                model="gemini-3.8-flash",
            )
            path = await store.asave(session)
            self.assertTrue(os.path.isfile(path))

            storage.append_turn(session, role="doctor", text="Wie geht es Ihnen?", source="human")
            await store.asave(session)

            reloaded = await store.aload("sess-local-1")
            self.assertEqual(len(reloaded["turns"]), 1)
            self.assertEqual(reloaded["turns"][0]["role"], "doctor")
            self.assertEqual(reloaded["turns"][0]["text"], "Wie geht es Ihnen?")

            storage.mark_session_ended(reloaded)
            await store.asave(reloaded)
            final = await store.aload("sess-local-1")
            self.assertEqual(final["status"], "ended_by_doctor")

            with self.assertRaises(storage.SessionNotFoundError):
                await store.aload("missing-session")


class TestGCSSessionStore(unittest.TestCase):
    def test_save_overwrite_and_load_gcs(self) -> None:
        fake_gcs = _InMemoryGCSClient()
        store = storage.GCSSessionStore(
            "my-chat-bucket",
            prefix="/chats/",
            client=fake_gcs,
        )
        self.assertEqual(store.backend_name, "gcs")
        self.assertEqual(
            store.describe_location("sess-123"),
            "gs://my-chat-bucket/chats/sess-123.json",
        )

        session = storage.create_session_record(
            session_id="sess-123",
            doctor_email="doctor@example.com",
            doctor_user_id="109876543210",
            ehr_id="sarah-levi-brca1",
            profile_id=2,
            profile_text="Profil 2",
            model="gemini-3.8-flash",
            feed_question="Is this normal for BRCA1?",
        )
        uri = store.save(session)
        self.assertEqual(uri, "gs://my-chat-bucket/chats/sess-123.json")

        blob = fake_gcs.buckets["my-chat-bucket"].blobs["chats/sess-123.json"]
        self.assertEqual(blob.last_content_type, "application/json")
        self.assertEqual(
            blob.metadata,
            {
                "ehr_id": "sarah-levi-brca1",
                "profile_id": "2",
                "feed_question": "Is this normal for BRCA1?",
            },
        )

        # Simulate per-turn overwrite
        storage.append_turn(session, role="doctor", text="Guten Tag!", source="human")
        store.save(session)
        reloaded = store.load("sess-123")
        self.assertEqual(len(reloaded["turns"]), 1)
        self.assertEqual(reloaded["turns"][0]["text"], "Guten Tag!")
        self.assertEqual(reloaded["metadata"]["feed_question"], "Is this normal for BRCA1?")
        self.assertEqual(blob.metadata["feed_question"], "Is this normal for BRCA1?")

        # Verify valid UTF-8 JSON stored in GCS
        raw = fake_gcs.buckets["my-chat-bucket"].objects["chats/sess-123.json"]
        parsed = json.loads(raw)
        self.assertEqual(parsed["session_id"], "sess-123")

        with self.assertRaises(storage.SessionNotFoundError):
            store.load("does-not-exist")

    def test_factory_selects_gcs_or_local_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"GCS_BUCKET": "", "LOCAL_STORAGE_DIR": tmpdir},
                clear=False,
            ):
                local_store = storage.create_session_store()
            self.assertIsInstance(local_store, storage.LocalSessionStore)

        fake_gcs = _InMemoryGCSClient()
        with patch.dict(
            os.environ,
            {"GCS_BUCKET": "prod-bucket", "GCS_CHAT_PREFIX": "golden_chats"},
            clear=False,
        ):
            gcs_store = storage.create_session_store(gcs_client=fake_gcs)
        self.assertIsInstance(gcs_store, storage.GCSSessionStore)
        self.assertEqual(
            gcs_store.describe_location("abc"),
            "gs://prod-bucket/golden_chats/abc.json",
        )


if __name__ == "__main__":
    unittest.main()
