"""Unit tests for app.auth."""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

_SIMULATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIMULATIONS_DIR not in sys.path:
    sys.path.insert(0, _SIMULATIONS_DIR)


def _install_optional_dependency_stubs() -> None:
    """Installs minimal stubs for fastapi and google-auth when absent in test env."""
    if "fastapi" not in sys.modules:
        try:
            importlib.import_module("fastapi")
        except ModuleNotFoundError:
            fastapi_mod = types.ModuleType("fastapi")

            class _HTTPException(Exception):
                def __init__(self, status_code: int, detail: str = "") -> None:
                    super().__init__(detail)
                    self.status_code = status_code
                    self.detail = detail

            class _Request:
                def __init__(self, headers: dict[str, str] | None = None) -> None:
                    self.headers = headers or {}

            status_ns = types.SimpleNamespace(HTTP_401_UNAUTHORIZED=401)
            fastapi_mod.HTTPException = _HTTPException
            fastapi_mod.Request = _Request
            fastapi_mod.status = status_ns
            sys.modules["fastapi"] = fastapi_mod

    if "google.oauth2.id_token" not in sys.modules:
        try:
            importlib.import_module("google.oauth2.id_token")
        except ModuleNotFoundError:
            google_pkg = sys.modules.get("google") or types.ModuleType("google")
            oauth2_pkg = types.ModuleType("google.oauth2")
            id_token_mod = types.ModuleType("google.oauth2.id_token")
            id_token_mod.verify_token = MagicMock()
            oauth2_pkg.id_token = id_token_mod
            google_pkg.oauth2 = oauth2_pkg

            auth_pkg = types.ModuleType("google.auth")
            transport_pkg = types.ModuleType("google.auth.transport")
            requests_mod = types.ModuleType("google.auth.transport.requests")
            requests_mod.Request = MagicMock
            transport_pkg.requests = requests_mod
            auth_pkg.transport = transport_pkg
            google_pkg.auth = auth_pkg

            sys.modules["google"] = google_pkg
            sys.modules["google.oauth2"] = oauth2_pkg
            sys.modules["google.oauth2.id_token"] = id_token_mod
            sys.modules["google.auth"] = auth_pkg
            sys.modules["google.auth.transport"] = transport_pkg
            sys.modules["google.auth.transport.requests"] = requests_mod


_install_optional_dependency_stubs()

from fastapi import HTTPException  # noqa: E402
from google.oauth2 import id_token  # noqa: E402

from app import auth  # noqa: E402


class TestIapAuthHelpers(unittest.TestCase):
    def test_strip_iap_prefix(self) -> None:
        self.assertEqual(
            auth.strip_iap_prefix("accounts.google.com:doctor@example.com"),
            "doctor@example.com",
        )
        self.assertEqual(
            auth.strip_iap_prefix("accounts.google.com:109876543210"),
            "109876543210",
        )
        self.assertEqual(auth.strip_iap_prefix("doctor@example.com"), "doctor@example.com")
        self.assertEqual(auth.strip_iap_prefix(None), "")

    def test_is_local_dev_auth_enabled_defaults_false(self) -> None:
        self.assertFalse(auth.is_local_dev_auth_enabled({}))
        self.assertFalse(auth.is_local_dev_auth_enabled({"ALLOW_LOCAL_DEV_AUTH": "false"}))
        self.assertFalse(auth.is_local_dev_auth_enabled({"ALLOW_LOCAL_DEV_AUTH": "0"}))
        self.assertTrue(auth.is_local_dev_auth_enabled({"ALLOW_LOCAL_DEV_AUTH": "true"}))
        self.assertTrue(auth.is_local_dev_auth_enabled({"ALLOW_LOCAL_DEV_AUTH": "1"}))


class TestAuthenticateHeaders(unittest.TestCase):
    def test_valid_iap_jwt_returns_authenticated_user(self) -> None:
        claims = {
            "iss": "https://cloud.google.com/iap",
            "email": "doctor@example.com",
            "sub": "accounts.google.com:109876543210",
        }
        with patch.object(id_token, "verify_token", return_value=claims) as mock_verify:
            user = auth.authenticate_headers(
                {"X-Goog-IAP-JWT-Assertion": "valid.jwt.token"},
                iap_audience="/projects/123456/global/backendServices/999",
                allow_local_dev_auth=False,
            )

        self.assertEqual(user.email, "doctor@example.com")
        self.assertEqual(user.user_id, "109876543210")
        self.assertEqual(user.auth_source, "iap_jwt")
        self.assertEqual(
            user.to_doctor_dict(),
            {"email": "doctor@example.com", "user_id": "109876543210"},
        )
        mock_verify.assert_called_once()
        _, kwargs = mock_verify.call_args
        self.assertEqual(kwargs["audience"], "/projects/123456/global/backendServices/999")
        self.assertEqual(kwargs["certs_url"], auth.IAP_CERTS_URL)

    def test_invalid_iap_jwt_raises_401(self) -> None:
        with patch.object(id_token, "verify_token", side_effect=ValueError("bad sig")):
            with self.assertRaises(HTTPException) as ctx:
                auth.authenticate_headers(
                    {"X-Goog-IAP-JWT-Assertion": "tampered.jwt"},
                    iap_audience="/projects/123456/global/backendServices/999",
                    allow_local_dev_auth=False,
                )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_wrong_issuer_raises_401(self) -> None:
        claims = {
            "iss": "https://evil.example.com",
            "email": "doctor@example.com",
            "sub": "109876543210",
        }
        with patch.object(id_token, "verify_token", return_value=claims):
            with self.assertRaises(HTTPException) as ctx:
                auth.authenticate_headers(
                    {"X-Goog-IAP-JWT-Assertion": "wrong-iss.jwt"},
                    iap_audience="/projects/123456/global/backendServices/999",
                    allow_local_dev_auth=False,
                )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_missing_email_or_sub_claim_raises_401(self) -> None:
        claims = {
            "iss": "https://cloud.google.com/iap",
            "email": "doctor@example.com",
        }
        with patch.object(id_token, "verify_token", return_value=claims):
            with self.assertRaises(HTTPException) as ctx:
                auth.authenticate_headers(
                    {"X-Goog-IAP-JWT-Assertion": "missing-sub.jwt"},
                    iap_audience="/projects/123456/global/backendServices/999",
                    allow_local_dev_auth=False,
                )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_missing_jwt_with_iap_audience_configured_raises_401(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            auth.authenticate_headers(
                {"X-Goog-Authenticated-User-Email": "accounts.google.com:spoof@example.com"},
                iap_audience="/projects/123456/global/backendServices/999",
                allow_local_dev_auth=False,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_unconfigured_auth_rejects_even_with_spoofed_headers(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            auth.authenticate_headers(
                {
                    "X-Goog-Authenticated-User-Email": "accounts.google.com:spoof@example.com",
                    "X-Goog-Authenticated-User-Id": "accounts.google.com:999",
                },
                iap_audience="",
                allow_local_dev_auth=False,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_local_dev_auth_fallback_uses_dev_user_email(self) -> None:
        user = auth.authenticate_headers(
            {},
            iap_audience="",
            allow_local_dev_auth=True,
            dev_user_email="dr.local@hospital.test",
            dev_user_id="dev-42",
        )
        self.assertEqual(user.email, "dr.local@hospital.test")
        self.assertEqual(user.user_id, "dev-42")
        self.assertEqual(user.auth_source, "local_dev")

    def test_get_current_user_reads_env_and_request_headers(self) -> None:
        fake_request = types.SimpleNamespace(headers={})
        with patch.dict(
            os.environ,
            {
                "IAP_AUDIENCE": "",
                "ALLOW_LOCAL_DEV_AUTH": "true",
                "DEV_USER_EMAIL": "dev-doctor@local",
            },
            clear=False,
        ):
            user = auth.get_current_user(fake_request)
        self.assertEqual(user.email, "dev-doctor@local")
        self.assertEqual(user.auth_source, "local_dev")


if __name__ == "__main__":
    unittest.main()
