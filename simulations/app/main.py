"""FastAPI application exposing REST endpoints and static chat UI."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.auth import AuthenticatedUser, get_current_user
from app.catalog import PatientCatalog
from app.config import Settings
from app.patient_engine import PatientEngine
from app.storage import (
    SESSION_STATUS_ACTIVE,
    SessionNotFoundError,
    SessionStore,
    append_turn,
    create_session_record,
    create_session_store,
    mark_session_ended,
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class CreateSessionRequest(BaseModel):
    ehr_id: str = Field(..., min_length=1)
    profile_id: int = Field(0, ge=0)
    language: str | None = None
    use_seed_question: bool = True


class SendMessageRequest(BaseModel):
    text: str = Field(..., min_length=1)


def _verify_session_owner(session: dict[str, Any], user: AuthenticatedUser) -> None:
    owner_email = str(session.get("doctor", {}).get("email", "")).strip().lower()
    if owner_email != user.email.strip().lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this chat session.",
        )


async def _load_owned_session(
    store: SessionStore,
    session_id: str,
    user: AuthenticatedUser,
) -> dict[str, Any]:
    try:
        session = await store.aload(session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        ) from exc

    _verify_session_owner(session, user)
    return session


def create_app(
    *,
    settings: Settings | None = None,
    catalog: PatientCatalog | None = None,
    store: SessionStore | None = None,
    engine: PatientEngine | None = None,
) -> FastAPI:
    """Creates and configures the FastAPI application."""
    resolved_settings = settings or Settings.from_env()
    resolved_catalog = catalog or PatientCatalog(
        data_dir=resolved_settings.patient_data_dir,
        gcs_data_prefix=resolved_settings.gcs_data_prefix or None,
        gcs_bucket=resolved_settings.gcs_bucket or None,
    )
    resolved_store = store or create_session_store(
        gcs_bucket=resolved_settings.gcs_bucket,
        gcs_chat_prefix=resolved_settings.gcs_chat_prefix,
        local_storage_dir=resolved_settings.local_storage_dir,
    )
    resolved_engine = engine or PatientEngine(
        project=resolved_settings.gcp_project,
        location=resolved_settings.gcp_location,
        model=resolved_settings.model_name,
        threshold=resolved_settings.replica_threshold,
        default_language=resolved_settings.default_language,
    )

    application = FastAPI(
        title="Simulated Patient Chat",
        description="Interactive chat UI for doctors to collect ground-truth clinical dialogues.",
    )
    application.state.settings = resolved_settings
    application.state.catalog = resolved_catalog
    application.state.store = resolved_store
    application.state.engine = resolved_engine

    if os.path.isdir(STATIC_DIR):
        application.mount(
            "/static",
            StaticFiles(directory=STATIC_DIR),
            name="static",
        )

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "storage_backend": resolved_store.backend_name,
        }

    @application.get("/")
    async def serve_index() -> FileResponse:
        index_path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.isfile(index_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat UI static/index.html is not installed yet.",
            )
        return FileResponse(index_path)

    @application.get("/api/me")
    async def get_me(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> dict[str, str]:
        return {
            "email": user.email,
            "user_id": user.user_id,
            "storage_backend": resolved_store.backend_name,
        }

    @application.get("/api/catalog")
    async def get_catalog(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        _ = user
        return {
            "patients": resolved_catalog.list_catalog(),
        }

    @application.post("/api/sessions", status_code=status.HTTP_201_CREATED)
    async def create_session(
        payload: CreateSessionRequest,
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        try:
            ehr_text = resolved_catalog.get_ehr_text(payload.ehr_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown patient ehr_id: {payload.ehr_id!r}",
            ) from exc

        try:
            profile_text = resolved_catalog.get_profile_text(
                payload.ehr_id, payload.profile_id
            )
        except IndexError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        language = (payload.language or resolved_settings.default_language).strip()
        seed_question = (
            resolved_catalog.sample_seed_question(payload.ehr_id)
            if payload.use_seed_question
            else None
        )
        session = create_session_record(
            doctor_email=user.email,
            doctor_user_id=user.user_id,
            ehr_id=payload.ehr_id,
            profile_id=payload.profile_id,
            profile_text=profile_text,
            model=resolved_settings.model_name,
            language=language,
            replica_threshold=resolved_settings.replica_threshold,
            max_turns=resolved_settings.max_turns,
            feed_question=seed_question,
        )

        opening = await resolved_engine.generate_opening_reply(
            profile=profile_text,
            ehr=ehr_text,
            seed_question=seed_question,
            language=language,
            model=resolved_settings.model_name,
            threshold=resolved_settings.replica_threshold,
        )
        turn_metadata: dict[str, Any] = {
            "char_count": opening["char_count"],
            "regenerated": opening["regenerated"],
        }
        if seed_question:
            turn_metadata["feed_question"] = seed_question
        append_turn(
            session,
            role="patient",
            text=opening["text"],
            source="llm",
            metadata=turn_metadata,
        )

        await resolved_store.asave(session)
        return session

    @application.get("/api/sessions/{session_id}")
    async def get_session(
        session_id: str,
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        return await _load_owned_session(resolved_store, session_id, user)

    @application.post("/api/sessions/{session_id}/messages")
    async def post_message(
        session_id: str,
        payload: SendMessageRequest,
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        cleaned_text = payload.text.strip()
        if not cleaned_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message text must not be empty.",
            )

        session = await _load_owned_session(resolved_store, session_id, user)
        if session.get("status") != SESSION_STATUS_ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Chat session has already ended.",
            )

        config = session.get("config", {})
        max_turns = int(config.get("max_turns", resolved_settings.max_turns))
        if len(session.get("turns", [])) >= max_turns:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Chat session reached maximum turn limit ({max_turns}).",
            )

        ehr_id = session["patient"]["ehr_id"]
        try:
            ehr_text = resolved_catalog.get_ehr_text(ehr_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Patient EHR '{ehr_id}' is no longer available in catalog.",
            ) from exc

        # Persist doctor ground-truth message BEFORE calling Gemini so human
        # input is never lost on LLM error or timeout.
        doctor_turn = append_turn(
            session,
            role="doctor",
            text=cleaned_text,
            source="human",
        )
        await resolved_store.asave(session)

        reply = await resolved_engine.generate_reply(
            profile=session["patient"]["profile_text"],
            ehr=ehr_text,
            turns=session["turns"],
            language=config.get("language", resolved_settings.default_language),
            model=config.get("model", resolved_settings.model_name),
            threshold=int(
                config.get("replica_threshold", resolved_settings.replica_threshold)
            ),
        )
        patient_turn = append_turn(
            session,
            role="patient",
            text=reply["text"],
            source="llm",
            metadata={
                "char_count": reply["char_count"],
                "regenerated": reply["regenerated"],
            },
        )
        await resolved_store.asave(session)

        return {
            "session_id": session["session_id"],
            "doctor_turn": doctor_turn,
            "patient_turn": patient_turn,
            "status": session["status"],
            "turn_count": len(session["turns"]),
            "updated_at": session["updated_at"],
        }

    @application.post("/api/sessions/{session_id}/end")
    async def end_session_route(
        session_id: str,
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        session = await _load_owned_session(resolved_store, session_id, user)
        mark_session_ended(session)
        await resolved_store.asave(session)
        return {
            "session_id": session["session_id"],
            "status": session["status"],
            "updated_at": session["updated_at"],
            "turn_count": len(session.get("turns", [])),
        }

    return application


app = create_app()
