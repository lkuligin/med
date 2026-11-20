"""Unit tests for app.patient_engine."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure simulations/ is on sys.path when running from any working directory.
_SIMULATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIMULATIONS_DIR not in sys.path:
    sys.path.insert(0, _SIMULATIONS_DIR)


def _install_optional_dependency_stubs() -> None:
    """Installs minimal stubs for google.genai and httpx when absent in test env."""
    if "httpx" not in sys.modules:
        try:
            importlib.import_module("httpx")
        except ModuleNotFoundError:
            httpx_mod = types.ModuleType("httpx")
            httpx_mod.AsyncClient = MagicMock
            sys.modules["httpx"] = httpx_mod

    if "google.genai" not in sys.modules:
        try:
            importlib.import_module("google.genai")
        except ModuleNotFoundError:
            google_pkg = sys.modules.get("google") or types.ModuleType("google")
            genai_mod = types.ModuleType("google.genai")
            genai_types = types.ModuleType("google.genai.types")

            class _FakeConfig:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            class _FakeHttpOptions:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            genai_types.GenerateContentConfig = _FakeConfig
            genai_types.HttpOptions = _FakeHttpOptions
            genai_mod.types = genai_types
            genai_mod.Client = MagicMock
            google_pkg.genai = genai_mod

            sys.modules["google"] = google_pkg
            sys.modules["google.genai"] = genai_mod
            sys.modules["google.genai.types"] = genai_types


_install_optional_dependency_stubs()

from app import patient_engine  # noqa: E402


def _make_mock_client(responses: list[str]) -> MagicMock:
    """Creates a mock google-genai Client whose aio.models.generate_content yields responses."""
    client = MagicMock()
    side_effects = [types.SimpleNamespace(text=text) for text in responses]
    client.aio.models.generate_content = AsyncMock(side_effect=side_effects)
    return client


class TestPatientEnginePromptsAndFormatting(unittest.TestCase):
    def test_prompts_exclude_fertig_stop_token_rule(self) -> None:
        self.assertNotIn("fertig", patient_engine.INTERACTIVE_PATIENT_SYSTEM_PROMPT.lower())
        self.assertNotIn("fertig", patient_engine.INTERACTIVE_PATIENT_TURN_PROMPT.lower())
        self.assertIn("1 bis 3 kurze Sätze", patient_engine.INTERACTIVE_PATIENT_SYSTEM_PROMPT)
        self.assertIn("1–3 Sätze", patient_engine.INTERACTIVE_PATIENT_TURN_PROMPT)

    def test_prompts_require_german_replies(self) -> None:
        self.assertIn("ausschließlich auf Deutsch", patient_engine.INTERACTIVE_PATIENT_SYSTEM_PROMPT)
        self.assertIn("ausschließlich auf Deutsch", patient_engine.INTERACTIVE_PATIENT_TURN_PROMPT)

    def test_prompts_address_doctor_and_allow_natural_greeting(self) -> None:
        self.assertIn("Arzt / einer Ärztin", patient_engine.INTERACTIVE_PATIENT_SYSTEM_PROMPT)
        self.assertNotIn("MAIA", patient_engine.INTERACTIVE_PATIENT_SYSTEM_PROMPT)
        self.assertNotIn("Chatbot", patient_engine.INTERACTIVE_PATIENT_SYSTEM_PROMPT)
        self.assertNotIn("Chatbot", patient_engine.INTERACTIVE_PATIENT_TURN_PROMPT)
        self.assertIn("Begrüßung", patient_engine.INTERACTIVE_PATIENT_SYSTEM_PROMPT)

    def test_build_system_instruction_embeds_profile_and_ehr(self) -> None:
        instruction = patient_engine.build_system_instruction(
            profile="Ängstlich, direkt",
            ehr='{"resourceType": "Bundle"}',
        )
        self.assertIn("Ängstlich, direkt", instruction)
        self.assertIn('{"resourceType": "Bundle"}', instruction)

    def test_format_history_empty_returns_start_marker(self) -> None:
        self.assertEqual(patient_engine.format_history(None), "START A NEW CHAT")
        self.assertEqual(patient_engine.format_history([]), "START A NEW CHAT")

    def test_format_history_maps_doctor_to_chatbot_and_patient_to_patient(self) -> None:
        turns = [
            {"role": "patient", "text": "Mir ist oft übel."},
            {"role": "doctor", "text": "Seit wann haben Sie die Übelkeit?"},
            {"role": "patient", "text": "Seit ungefähr drei Tagen."},
        ]
        formatted = patient_engine.format_history(turns)
        self.assertEqual(
            formatted,
            "Patient: Mir ist oft übel.\n"
            "Chatbot: Seit wann haben Sie die Übelkeit?\n"
            "Patient: Seit ungefähr drei Tagen.",
        )

    def test_load_ehr_hint_delegates_to_create_ehr_hint(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump({"resourceType": "Patient", "id": "p1"}, tmp)
            tmp_path = tmp.name
        try:
            hint = patient_engine.load_ehr_hint(tmp_path)
            self.assertIn('"resourceType": "Patient"', hint)
            self.assertIn('"id": "p1"', hint)
        finally:
            os.unlink(tmp_path)


class TestPatientEngineGeneration(unittest.IsolatedAsyncioTestCase):
    async def test_generate_short_reply_cleans_prefix_without_regeneration(self) -> None:
        client = _make_mock_client(['Patient: "Eigentlich fast durchgehend seit einem Monat."'])
        engine = patient_engine.PatientEngine(client=client)

        result = await engine.generate_reply(
            profile="Skeptisch, kurz angebunden",
            ehr='{"id": "ehr-1"}',
            turns=[
                {"role": "patient", "text": "Hallo, ich bin ständig erschöpft."},
                {"role": "doctor", "text": "Seit wann ist die Müdigkeit besonders stark?"},
            ],
        )

        self.assertEqual(
            result,
            {
                "text": "Eigentlich fast durchgehend seit einem Monat.",
                "char_count": len("Eigentlich fast durchgehend seit einem Monat."),
                "regenerated": False,
            },
        )
        self.assertEqual(client.aio.models.generate_content.await_count, 1)
        call_kwargs = client.aio.models.generate_content.await_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.8-flash")
        self.assertEqual(call_kwargs["config"].temperature, 0.7)
        self.assertIn("Chatbot: Seit wann ist die Müdigkeit besonders stark?", call_kwargs["contents"])

    async def test_generate_long_reply_triggers_regeneration(self) -> None:
        long_reply = "A" * 350
        short_reply = "Das macht mir gerade wirklich Sorgen. Kann ich das Medikament weiternehmen?"
        client = _make_mock_client([long_reply, short_reply])
        engine = patient_engine.PatientEngine(client=client, threshold=300)

        result = await engine.generate_reply(
            profile="Besorgt",
            ehr='{"id": "ehr-2"}',
            turns=[{"role": "doctor", "text": "Wie geht es Ihnen heute?"}],
            language="de",
        )

        self.assertEqual(result["text"], short_reply)
        self.assertEqual(result["char_count"], len(short_reply))
        self.assertTrue(result["regenerated"])
        self.assertEqual(client.aio.models.generate_content.await_count, 2)

    async def test_generate_opening_reply_uses_start_new_chat(self) -> None:
        client = _make_mock_client(["Hallo, ich habe eine Frage zu meinen Tabletten."])
        engine = patient_engine.PatientEngine(client=client)

        result = await engine.generate_opening_reply(
            profile="Offen",
            ehr='{"id": "ehr-3"}',
        )

        self.assertEqual(result["text"], "Hallo, ich habe eine Frage zu meinen Tabletten.")
        self.assertFalse(result["regenerated"])
        call_kwargs = client.aio.models.generate_content.await_args.kwargs
        self.assertIn("START A NEW CHAT", call_kwargs["contents"])

    async def test_generate_opening_reply_with_seed_question_uses_opening_prompt(self) -> None:
        client = _make_mock_client(["Guten Tag, ich mache mir große Sorgen wegen meiner Zyste."])
        engine = patient_engine.PatientEngine(client=client)

        result = await engine.generate_opening_reply(
            profile="Ängstlich",
            ehr='{"id": "ehr-4"}',
            seed_question="The ultrasound report says complex cyst. Do I need surgery?",
        )

        self.assertEqual(
            result["text"], "Guten Tag, ich mache mir große Sorgen wegen meiner Zyste."
        )
        self.assertFalse(result["regenerated"])
        call_kwargs = client.aio.models.generate_content.await_args.kwargs
        self.assertIn(
            "The ultrasound report says complex cyst. Do I need surgery?",
            call_kwargs["contents"],
        )
        self.assertIn("Begrüßung", call_kwargs["contents"])
        self.assertIn("ausschließlich auf Deutsch", call_kwargs["contents"])

    async def test_lazy_client_creation_uses_vertexai_config(self) -> None:
        fake_client = _make_mock_client(["Danke, das hilft mir weiter."])
        with patch.object(patient_engine, "create_genai_client", return_value=fake_client) as factory:
            engine = patient_engine.PatientEngine(project="my-test-project", location="global")
            factory.assert_not_called()

            result = await engine.generate_reply(
                profile="Freundlich",
                ehr="{}",
                turns=[{"role": "doctor", "text": "Haben Sie noch Fragen?"}],
            )

            factory.assert_called_once_with(project="my-test-project", location="global")
            self.assertEqual(result["text"], "Danke, das hilft mir weiter.")


if __name__ == "__main__":
    unittest.main()
