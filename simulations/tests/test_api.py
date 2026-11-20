"""Unit tests for app.config and app.main REST API endpoints."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, MagicMock

_SIMULATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIMULATIONS_DIR not in sys.path:
    sys.path.insert(0, _SIMULATIONS_DIR)


def _install_optional_dependency_stubs() -> None:
    """Installs minimal stubs for fastapi, pydantic, httpx, and google.genai when absent."""
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
            genai_types.GenerateContentConfig = lambda **kw: types.SimpleNamespace(**kw)
            genai_types.HttpOptions = lambda **kw: types.SimpleNamespace(**kw)
            genai_mod.types = genai_types
            genai_mod.Client = MagicMock
            google_pkg.genai = genai_mod
            sys.modules["google"] = google_pkg
            sys.modules["google.genai"] = genai_mod
            sys.modules["google.genai.types"] = genai_types

    if "pydantic" not in sys.modules:
        try:
            importlib.import_module("pydantic")
        except ModuleNotFoundError:
            pydantic_mod = types.ModuleType("pydantic")

            class _BaseModel:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
                    for name, default in self.__class__.__dict__.items():
                        if (
                            not name.startswith("_")
                            and not callable(default)
                            and not hasattr(self, name)
                        ):
                            setattr(self, name, default)

            def _Field(default=..., **_kwargs):
                return None if default is ... else default

            pydantic_mod.BaseModel = _BaseModel
            pydantic_mod.Field = _Field
            sys.modules["pydantic"] = pydantic_mod

    if "fastapi" not in sys.modules:
        try:
            importlib.import_module("fastapi")
        except ModuleNotFoundError:
            fastapi_mod = types.ModuleType("fastapi")
            responses_mod = types.ModuleType("fastapi.responses")
            staticfiles_mod = types.ModuleType("fastapi.staticfiles")

            class _HTTPException(Exception):
                def __init__(self, status_code: int, detail: str = "") -> None:
                    super().__init__(detail)
                    self.status_code = status_code
                    self.detail = detail

            class _Request:
                def __init__(self, headers: dict[str, str] | None = None) -> None:
                    self.headers = headers or {}

            class _FastAPI:
                def __init__(self, **_kwargs) -> None:
                    self.state = types.SimpleNamespace()
                    self.routes_map: dict[tuple[str, str], object] = {}
                    self.mounts: list[tuple[str, object]] = []

                def mount(self, path: str, app_obj: object, name: str = "") -> None:
                    self.mounts.append((path, app_obj))

                def _register(self, method: str, path: str):
                    def decorator(fn):
                        self.routes_map[(method, path)] = fn
                        return fn

                    return decorator

                def get(self, path: str, **_kwargs):
                    return self._register("GET", path)

                def post(self, path: str, **_kwargs):
                    return self._register("POST", path)

            class _FileResponse:
                def __init__(self, path: str) -> None:
                    self.path = path

            class _StaticFiles:
                def __init__(self, directory: str) -> None:
                    self.directory = directory

            status_ns = types.SimpleNamespace(
                HTTP_200_OK=200,
                HTTP_201_CREATED=201,
                HTTP_400_BAD_REQUEST=400,
                HTTP_401_UNAUTHORIZED=401,
                HTTP_403_FORBIDDEN=403,
                HTTP_404_NOT_FOUND=404,
                HTTP_409_CONFLICT=409,
                HTTP_500_INTERNAL_SERVER_ERROR=500,
            )

            fastapi_mod.Depends = lambda dep: dep
            fastapi_mod.FastAPI = _FastAPI
            fastapi_mod.HTTPException = _HTTPException
            fastapi_mod.Request = _Request
            fastapi_mod.status = status_ns
            responses_mod.FileResponse = _FileResponse
            staticfiles_mod.StaticFiles = _StaticFiles

            sys.modules["fastapi"] = fastapi_mod
            sys.modules["fastapi.responses"] = responses_mod
            sys.modules["fastapi.staticfiles"] = staticfiles_mod
    else:
        # Ensure sub-modules and extra status codes exist if test_auth ran first.
        fastapi_mod = sys.modules["fastapi"]
        if not hasattr(fastapi_mod, "FastAPI"):
            del sys.modules["fastapi"]
            _install_optional_dependency_stubs()


_install_optional_dependency_stubs()

from fastapi import HTTPException  # noqa: E402

from app.auth import AuthenticatedUser  # noqa: E402
from app.catalog import PatientCatalog  # noqa: E402
from app.config import Settings  # noqa: E402
from app import main as app_main  # noqa: E402
from app.storage import LocalSessionStore  # noqa: E402


def _get_route_handler(application, method: str, path: str):
    if hasattr(application, "routes_map"):
        return application.routes_map[(method, path)]
    for route in application.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise KeyError(f"Route not found: {method} {path}")


class TestSettings(unittest.TestCase):
    def test_from_env_parses_defaults_and_overrides(self) -> None:
        settings = Settings.from_env(
            {
                "GCP_PROJECT": "demo-proj",
                "MODEL_NAME": "gemini-3.8-flash",
                "REPLICA_THRESHOLD": "250",
                "MAX_TURNS": "40",
                "GCS_BUCKET": "demo-bucket",
                "GCS_CHAT_PREFIX": "golden",
            }
        )
        self.assertEqual(settings.gcp_project, "demo-proj")
        self.assertEqual(settings.replica_threshold, 250)
        self.assertEqual(settings.max_turns, 40)
        self.assertEqual(settings.gcs_bucket, "demo-bucket")
        self.assertEqual(settings.gcs_chat_prefix, "golden")


class TestRestApiEndpoints(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        data_dir = os.path.join(self._tmpdir.name, "data")
        storage_dir = os.path.join(self._tmpdir.name, "chats")
        os.makedirs(data_dir, exist_ok=True)

        with open(os.path.join(data_dir, "sarah-levi-brca1.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "resourceType": "Patient",
                    "name": [{"given": ["Sarah"], "family": "Levi"}],
                },
                f,
            )
        with open(
            os.path.join(data_dir, "profiles_sarah-levi-brca1.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"profiles": ["Profil 0 Text", "Profil 1 Text"]}, f)
        with open(
            os.path.join(data_dir, "sarah-levi-brca1_questions.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"questions": ["Hallo, ich bin ständig müde."]}, f)

        self.catalog = PatientCatalog(data_dir=data_dir, gcs_data_prefix="")
        self.store = LocalSessionStore(storage_dir)
        self.engine = MagicMock()
        self.engine.generate_opening_reply = AsyncMock(
            return_value={
                "text": "Guten Tag, ich habe Fragen zu meiner Behandlung.",
                "char_count": 47,
                "regenerated": False,
            }
        )
        self.engine.generate_reply = AsyncMock(
            return_value={
                "text": "Seit ungefähr vier Wochen fast jeden Tag.",
                "char_count": 41,
                "regenerated": False,
            }
        )

        self.settings = Settings(
            gcp_project="test-project",
            patient_data_dir=data_dir,
            local_storage_dir=storage_dir,
        )
        self.application = app_main.create_app(
            settings=self.settings,
            catalog=self.catalog,
            store=self.store,
            engine=self.engine,
        )
        self.doctor = AuthenticatedUser(email="doctor@example.com", user_id="1001")
        self.other_doctor = AuthenticatedUser(email="other@example.com", user_id="2002")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    async def test_healthz_me_and_catalog(self) -> None:
        health = await _get_route_handler(self.application, "GET", "/healthz")()
        self.assertEqual(health, {"status": "ok", "storage_backend": "local"})

        index_resp = await _get_route_handler(self.application, "GET", "/")()
        self.assertTrue(str(index_resp.path).endswith(os.path.join("static", "index.html")))

        me = await _get_route_handler(self.application, "GET", "/api/me")(user=self.doctor)
        self.assertEqual(me["email"], "doctor@example.com")
        self.assertEqual(me["user_id"], "1001")

        cat = await _get_route_handler(self.application, "GET", "/api/catalog")(user=self.doctor)
        self.assertEqual(len(cat["patients"]), 1)
        self.assertEqual(cat["patients"][0]["ehr_id"], "sarah-levi-brca1")

    async def test_full_session_flow_and_owner_check(self) -> None:
        create_fn = _get_route_handler(self.application, "POST", "/api/sessions")
        get_fn = _get_route_handler(self.application, "GET", "/api/sessions/{session_id}")
        msg_fn = _get_route_handler(
            self.application, "POST", "/api/sessions/{session_id}/messages"
        )
        end_fn = _get_route_handler(self.application, "POST", "/api/sessions/{session_id}/end")

        # 1. Create session using seed question
        session = await create_fn(
            payload=app_main.CreateSessionRequest(
                ehr_id="sarah-levi-brca1",
                profile_id=1,
                language="de",
                use_seed_question=True,
            ),
            user=self.doctor,
        )
        sid = session["session_id"]
        self.assertEqual(session["status"], "active")
        self.assertEqual(len(session["turns"]), 1)
        self.assertEqual(session["turns"][0]["source"], "llm")
        self.assertEqual(
            session["turns"][0]["text"],
            "Guten Tag, ich habe Fragen zu meiner Behandlung.",
        )
        self.assertEqual(
            session["turns"][0]["metadata"]["feed_question"],
            "Hallo, ich bin ständig müde.",
        )
        self.assertEqual(
            session["metadata"]["feed_question"],
            "Hallo, ich bin ständig müde.",
        )
        self.engine.generate_opening_reply.assert_awaited_once()
        self.assertEqual(
            self.engine.generate_opening_reply.await_args.kwargs["seed_question"],
            "Hallo, ich bin ständig müde.",
        )

        # 2. Reload session as owner succeeds; other doctor gets 403
        reloaded = await get_fn(session_id=sid, user=self.doctor)
        self.assertEqual(reloaded["session_id"], sid)
        with self.assertRaises(HTTPException) as ctx:
            await get_fn(session_id=sid, user=self.other_doctor)
        self.assertEqual(ctx.exception.status_code, 403)

        # 3. Post doctor message -> doctor + patient turns persisted
        msg_resp = await msg_fn(
            session_id=sid,
            payload=app_main.SendMessageRequest(text="Seit wann ist die Müdigkeit so stark?"),
            user=self.doctor,
        )
        self.assertEqual(msg_resp["doctor_turn"]["role"], "doctor")
        self.assertEqual(msg_resp["doctor_turn"]["source"], "human")
        self.assertEqual(msg_resp["patient_turn"]["role"], "patient")
        self.assertEqual(msg_resp["patient_turn"]["source"], "llm")
        self.assertEqual(msg_resp["turn_count"], 3)

        persisted = self.store.load(sid)
        self.assertEqual(len(persisted["turns"]), 3)

        # 4. End session -> status updated, further messages rejected with 409
        end_resp = await end_fn(session_id=sid, user=self.doctor)
        self.assertEqual(end_resp["status"], "ended_by_doctor")
        self.assertEqual(self.store.load(sid)["status"], "ended_by_doctor")

        with self.assertRaises(HTTPException) as ended_ctx:
            await msg_fn(
                session_id=sid,
                payload=app_main.SendMessageRequest(text="Noch eine Frage"),
                user=self.doctor,
            )
        self.assertEqual(ended_ctx.exception.status_code, 409)

    async def test_doctor_message_persisted_even_if_llm_fails(self) -> None:
        create_fn = _get_route_handler(self.application, "POST", "/api/sessions")
        msg_fn = _get_route_handler(
            self.application, "POST", "/api/sessions/{session_id}/messages"
        )

        session = await create_fn(
            payload=app_main.CreateSessionRequest(
                ehr_id="sarah-levi-brca1",
                profile_id=0,
                use_seed_question=False,
            ),
            user=self.doctor,
        )
        sid = session["session_id"]
        self.assertEqual(session["turns"][0]["source"], "llm")
        self.assertIsNone(session["metadata"]["feed_question"])

        self.engine.generate_reply = AsyncMock(side_effect=RuntimeError("Vertex AI timeout"))
        with self.assertRaises(RuntimeError):
            await msg_fn(
                session_id=sid,
                payload=app_main.SendMessageRequest(text="Wichtige ärztliche Antwort"),
                user=self.doctor,
            )

        # Doctor's human ground-truth turn must still be safely persisted!
        saved = self.store.load(sid)
        self.assertEqual(len(saved["turns"]), 2)
        self.assertEqual(saved["turns"][1]["role"], "doctor")
        self.assertEqual(saved["turns"][1]["text"], "Wichtige ärztliche Antwort")


if __name__ == "__main__":
    unittest.main()
