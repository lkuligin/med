"""Interactive patient replica generation engine.

Generates simulated patient replies using Vertex AI (Gemini) while a human
doctor writes the clinician responses. Reuses the existing replica cleaning,
length-checking, regeneration, history-formatting, and EHR-loading helpers
from ``_simulate_chat.py`` and ``_ehr.py`` without modifying them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any, TypedDict

from _ehr import create_ehr_hint
from _simulate_chat import (
    _DEFAULT_MODEL,
    _MAX_LENGTH,
    _REPLICA_THRESHOLD,
    _clean_replica,
    _is_replica_too_long,
    _merge_history,
    _regenerate_replica,
)
from google import genai
from google.genai import types

DEFAULT_MODEL = _DEFAULT_MODEL
DEFAULT_REPLICA_THRESHOLD = _REPLICA_THRESHOLD
DEFAULT_MAX_OUTPUT_TOKENS = _MAX_LENGTH
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_REGENERATION_RETRIES = 3
DEFAULT_LANGUAGE = "de"
DEFAULT_LOCATION = "global"

# Interactive prompts mirror _PROMPT_SIMULATE_STEP1 / _PROMPT_SIMULATE_STEP2
# from _prompts.py, minus the 'fertig' stop-token rule so the human doctor
# controls when the consultation ends and the patient gives natural closings.
INTERACTIVE_PATIENT_SYSTEM_PROMPT = """Du bist ein Krebspatient / eine Krebspatientin in einem Chatgespräch mit einem Arzt / einer Ärztin.

Dein psychologisches Profil:
{profile}

Deine elektronische Patientenakte (EHR):
{ehr}

Verhalte dich genau wie ein echter Mensch, der in einem Messenger/Chat auf seinem Smartphone oder Laptop mit seinem Arzt / seiner Ärztin schreibt:
1. SEHR KURZE ANTWORTEN (Chat-Stil):
   - Schreibe wie in einem echten Chat: in der Regel nur 1 bis 3 kurze Sätze (maximal ein kurzer Absatz).
   - Schreibe NIEMALS lange Abhandlungen, Essays oder gegliederte Listen.

2. KEINE WIEDERHOLUNG ODER ZUSAMMENFASSUNG DES ARZTES / DER ÄRZTIN:
   - Wiederhole oder paraphrasiere KEINESFALLS, was der Arzt / die Ärztin dir gerade geantwortet oder erklärt hat.
   - Fasse die Aussagen, Statistiken oder Ratschläge des Arztes / der Ärztin NICHT zusammen.
   - Beginne deine Nachricht NIEMALS mit Floskeln wie „Sie haben gesagt...“, „Dass die Datenlage zu X so ist...“, „Ihre Ausführungen zeigen...“, „Die Information, dass...“ oder Bestätigungen genannter Zahlen. Der Arzt / Die Ärztin weiß selbst, was er/sie geschrieben hat.

3. AUTHENTISCH UND MENSCHLICH KLINGEN:
   - Sprich in natürlicher Alltagssprache, direkt und ungekünstelt aus deiner persönlichen Patienten-Perspektive.
   - Wenn das Gespräch gerade erst beginnt oder der Arzt / die Ärztin dich begrüßt, erwidere die Begrüßung kurz und natürlich (z. B. „Hallo“ oder „Guten Tag“), bevor du auf das Thema eingehst.
   - Reagiere mit echten menschlichen Emotionen (z. B. Sorge, Skepsis, Erleichterung, Frustration, Verunsicherung oder Ungeduld), passend zu deinem Profil.
   - Klinge niemals wie eine KI, ein medizinisches Gutachten oder ein Prüfer.

4. FOKUS:
   - Reagiere nur auf den einen Gedanken, der dich gerade am meisten berührt oder interessiert, oder stelle direkt eine kurze Folgefrage.

5. SPRACHE (IMMER DEUTSCH):
   - Antworte IMMER ausschließlich auf Deutsch (natürliches, alltagssprachliches Deutsch), auch wenn dein Profil, deine Patientenakte (EHR) oder frühere Nachrichten im Chat auf Englisch oder in einer anderen Sprache verfasst sind."""

INTERACTIVE_PATIENT_TURN_PROMPT = """Bisheriger Gesprächsverlauf:
{history}

Wichtige Regeln für deine Antwort als Patient:
- Antworte IMMER ausschließlich auf Deutsch (auch wenn vorherige Nachrichten oder dein Profil auf Englisch sind).
- Antworte kurz (1–3 Sätze im Chat-Stil).
- Wenn das Gespräch gerade beginnt oder der Arzt / die Ärztin dich begrüßt hat, grüße kurz und natürlich zurück.
- Wiederhole und fasse NICHT zusammen, was der Arzt / die Ärztin gerade gesagt hat.
- Reagiere direkt aus deiner persönlichen Sicht oder stelle deine nächste konkrete Frage.

Deine nächste Nachricht als Patient (auf Deutsch):"""

INTERACTIVE_PATIENT_OPENING_PROMPT = """Du beginnst jetzt ein neues Chatgespräch mit deinem Arzt / deiner Ärztin.

Thematische Ausgangslage / Kernanliegen für deine erste Nachricht (als inhaltliche Basis):
{seed_question}

Wichtige Regeln für deine erste Nachricht als Patient:
- Antworte IMMER ausschließlich auf Deutsch (natürliches, alltagssprachliches Deutsch).
- Beginne mit einer kurzen, natürlichen Begrüßung (z. B. „Hallo,“ oder „Guten Tag,“).
- Nutze die obige Ausgangslage als inhaltliche und emotionale Basis, aber formuliere sie frei, authentisch und mit natürlicher menschlicher Variation in deinen eigenen Worten (passend zu deinem Profil) — übersetze sie NICHT mechanisch Wort für Wort.
- Antworte kurz (1–3 Sätze im Chat-Stil).

Deine erste Nachricht als Patient (auf Deutsch):"""


class PatientReplicaResult(TypedDict):
    """Structured result returned by the patient replica generator."""

    text: str
    char_count: int
    regenerated: bool


def load_ehr_hint(ehr_path: str) -> str:
    """Loads formatted EHR text using the shared ``_ehr.create_ehr_hint`` helper."""
    return create_ehr_hint(ehr_path)


def build_system_instruction(profile: str, ehr: str) -> str:
    """Builds the interactive patient system instruction (without 'fertig' rule)."""
    return INTERACTIVE_PATIENT_SYSTEM_PROMPT.format(profile=profile, ehr=ehr)


def format_history(turns: Sequence[Mapping[str, Any]] | None) -> str:
    """Formats session turns for the patient LLM using ``_simulate_chat._merge_history``.

    Stored ``doctor`` turns are explicitly mapped to the ``Chatbot:`` side so the
    patient LLM receives the exact same history format as in ``_simulate_chat.py``,
    while callers persist the real ``"doctor"`` speaker role in session storage.
    """
    if not turns:
        return "START A NEW CHAT"

    normalized: list[dict[str, str]] = []
    for turn in turns:
        raw_role = str(turn.get("role", "")).strip().lower()
        text = str(turn.get("text") or turn.get("content") or "")
        mapped_role = "chat" if raw_role == "doctor" else raw_role
        normalized.append({"role": mapped_role, "text": text})

    merged = _merge_history(normalized)
    return merged if merged else "START A NEW CHAT"


def build_turn_prompt(turns: Sequence[Mapping[str, Any]] | None) -> tuple[str, str]:
    """Builds ``(history_str, prompt_text)`` for the next patient turn."""
    history_str = format_history(turns)
    prompt_text = INTERACTIVE_PATIENT_TURN_PROMPT.format(history=history_str)
    return history_str, prompt_text


def build_opening_prompt(seed_question: str) -> tuple[str, str]:
    """Builds ``(history_str, prompt_text)`` for an opening turn based on a feed question."""
    history_str = "START A NEW CHAT"
    prompt_text = INTERACTIVE_PATIENT_OPENING_PROMPT.format(
        seed_question=seed_question.strip()
    )
    return history_str, prompt_text


def create_genai_client(
    project: str | None = None,
    location: str = DEFAULT_LOCATION,
) -> genai.Client:
    """Creates a Vertex AI ``google-genai`` client."""
    resolved_project = project or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    return genai.Client(
        http_options=types.HttpOptions(api_version="v1"),
        vertexai=True,
        project=resolved_project,
        location=location,
    )


async def generate_patient_replica(
    *,
    profile: str,
    ehr: str,
    client: genai.Client,
    turns: Sequence[Mapping[str, Any]] | None = None,
    seed_question: str | None = None,
    language: str = DEFAULT_LANGUAGE,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    threshold: int = DEFAULT_REPLICA_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_REGENERATION_RETRIES,
) -> PatientReplicaResult:
    """Generates a single simulated patient reply for an interactive doctor chat.

    Uses the interactive prompts (without the ``'fertig'`` stop-token rule),
    cleans the output with ``_clean_replica``, and enforces the character length
    threshold via ``_is_replica_too_long`` and ``_regenerate_replica``.
    """
    system_instruction = build_system_instruction(profile=profile, ehr=ehr)
    if seed_question and not turns:
        history_str, prompt_text = build_opening_prompt(seed_question)
    else:
        history_str, prompt_text = build_turn_prompt(turns)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt_text,
        config=config,
    )
    cleaned = _clean_replica(response.text or "")
    regenerated = False

    if _is_replica_too_long(cleaned, threshold):
        cleaned = await _regenerate_replica(
            client=client,
            previous_replica=cleaned,
            history_str=history_str,
            system_instruction=system_instruction,
            threshold=threshold,
            language=language,
            model=model,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
        )
        regenerated = True

    return {
        "text": cleaned,
        "char_count": len(cleaned),
        "regenerated": regenerated,
    }


class PatientEngine:
    """Stateful configuration wrapper around ``generate_patient_replica``."""

    def __init__(
        self,
        *,
        client: genai.Client | None = None,
        project: str | None = None,
        location: str = DEFAULT_LOCATION,
        model: str = DEFAULT_MODEL,
        threshold: int = DEFAULT_REPLICA_THRESHOLD,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_REGENERATION_RETRIES,
        default_language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self._client = client
        self._project = project
        self._location = location
        self.model = model
        self.threshold = threshold
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.default_language = default_language

    @property
    def client(self) -> genai.Client:
        """Returns the configured ``genai.Client``, creating it lazily if needed."""
        if self._client is None:
            self._client = create_genai_client(
                project=self._project,
                location=self._location,
            )
        return self._client

    async def generate_reply(
        self,
        *,
        profile: str,
        ehr: str,
        turns: Sequence[Mapping[str, Any]] | None = None,
        seed_question: str | None = None,
        language: str | None = None,
        model: str | None = None,
        threshold: int | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> PatientReplicaResult:
        """Generates the next simulated patient replica given prior conversation turns."""
        return await generate_patient_replica(
            profile=profile,
            ehr=ehr,
            client=self.client,
            turns=turns,
            seed_question=seed_question,
            language=language or self.default_language,
            model=model or self.model,
            max_output_tokens=(
                self.max_output_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
            threshold=self.threshold if threshold is None else threshold,
            temperature=self.temperature if temperature is None else temperature,
            max_retries=self.max_retries if max_retries is None else max_retries,
        )

    async def generate_opening_reply(
        self,
        *,
        profile: str,
        ehr: str,
        seed_question: str | None = None,
        language: str | None = None,
        model: str | None = None,
        threshold: int | None = None,
    ) -> PatientReplicaResult:
        """Generates an opening patient message, optionally grounded in a feed question."""
        return await self.generate_reply(
            profile=profile,
            ehr=ehr,
            turns=None,
            seed_question=seed_question,
            language=language,
            model=model,
            threshold=threshold,
        )
