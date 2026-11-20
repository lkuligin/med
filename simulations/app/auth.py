"""IAP JWT authentication and local development auth fallback.

Provides the FastAPI dependency ``get_current_user(request)``:
1. When ``IAP_AUDIENCE`` is configured (production), verifies the
   ``X-Goog-IAP-JWT-Assertion`` header against Google's IAP public certificates
   using ``google.oauth2.id_token.verify_token`` and extracts ``email`` and
   ``sub``.
2. When ``ALLOW_LOCAL_DEV_AUTH=true`` (local development only, default
   ``false``), falls back to ``DEV_USER_EMAIL`` (default ``dev-doctor@local``).
3. Rejects unauthenticated or invalid requests with HTTP 401 Unauthorized.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

from fastapi import HTTPException, Request, status

IAP_JWT_HEADER = "X-Goog-IAP-JWT-Assertion"
IAP_EMAIL_HEADER = "X-Goog-Authenticated-User-Email"
IAP_USER_ID_HEADER = "X-Goog-Authenticated-User-Id"

IAP_CERTS_URL = "https://www.gstatic.com/iap/verify/public_key"
IAP_ISSUER = "https://cloud.google.com/iap"

DEFAULT_DEV_USER_EMAIL = "dev-doctor@local"
DEFAULT_DEV_USER_ID = "local-dev-user"

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class AuthenticatedUser:
    """Identity of the authenticated doctor making a request."""

    email: str
    user_id: str
    auth_source: str = "iap_jwt"

    def to_doctor_dict(self) -> dict[str, str]:
        """Returns the ``doctor`` metadata structure persisted in session JSON."""
        return {
            "email": self.email,
            "user_id": self.user_id,
        }


def is_local_dev_auth_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Returns True only when ``ALLOW_LOCAL_DEV_AUTH`` is explicitly truthy."""
    source = os.environ if env is None else env
    raw = source.get("ALLOW_LOCAL_DEV_AUTH", "false").strip().lower()
    return raw in _TRUTHY_VALUES


def strip_iap_prefix(value: str | None) -> str:
    """Strips the ``accounts.google.com:`` prefix from IAP identity headers/claims."""
    if not value:
        return ""
    cleaned = value.strip()
    if ":" in cleaned and cleaned.lower().startswith("accounts.google.com:"):
        return cleaned.split(":", 1)[1].strip()
    return cleaned


def _get_header_case_insensitive(
    headers: Mapping[str, str], target_name: str
) -> str | None:
    """Reads an HTTP header value case-insensitively."""
    direct = headers.get(target_name)
    if direct is not None:
        return direct
    lowered = target_name.lower()
    direct_lower = headers.get(lowered)
    if direct_lower is not None:
        return direct_lower
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def verify_iap_jwt(
    jwt_token: str,
    expected_audience: str,
    *,
    certs_url: str = IAP_CERTS_URL,
) -> dict[str, Any]:
    """Verifies a Google Cloud IAP JWT assertion and returns its decoded claims."""
    if not jwt_token or not jwt_token.strip():
        raise ValueError("Missing IAP JWT assertion token.")
    if not expected_audience or not expected_audience.strip():
        raise ValueError("IAP_AUDIENCE is not configured.")

    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    claims = id_token.verify_token(
        jwt_token.strip(),
        google_requests.Request(),
        audience=expected_audience.strip(),
        certs_url=certs_url,
    )
    if not isinstance(claims, Mapping):
        raise ValueError("Verified IAP token did not return a claims mapping.")

    issuer = claims.get("iss")
    if issuer != IAP_ISSUER:
        raise ValueError(f"Unexpected IAP JWT issuer: {issuer!r}")

    return dict(claims)


def authenticate_headers(
    headers: Mapping[str, str],
    *,
    iap_audience: str | None = None,
    allow_local_dev_auth: bool | None = None,
    dev_user_email: str | None = None,
    dev_user_id: str | None = None,
) -> AuthenticatedUser:
    """Authenticates a request from its HTTP headers and environment configuration.

    Raises ``HTTPException(401)`` when authentication fails.
    """
    resolved_audience = (
        os.getenv("IAP_AUDIENCE", "").strip()
        if iap_audience is None
        else iap_audience.strip()
    )
    resolved_dev_enabled = (
        is_local_dev_auth_enabled()
        if allow_local_dev_auth is None
        else allow_local_dev_auth
    )

    jwt_token = _get_header_case_insensitive(headers, IAP_JWT_HEADER)

    if resolved_audience:
        if jwt_token:
            try:
                claims = verify_iap_jwt(jwt_token, resolved_audience)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid IAP JWT assertion.",
                ) from exc

            email = strip_iap_prefix(str(claims.get("email") or ""))
            sub = strip_iap_prefix(str(claims.get("sub") or ""))
            if not email or not sub:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="IAP JWT assertion is missing required identity claims.",
                )
            return AuthenticatedUser(
                email=email,
                user_id=sub,
                auth_source="iap_jwt",
            )

        if not resolved_dev_enabled:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing IAP authentication header.",
            )

    if resolved_dev_enabled:
        resolved_email = (
            os.getenv("DEV_USER_EMAIL", DEFAULT_DEV_USER_EMAIL).strip()
            if dev_user_email is None
            else dev_user_email.strip()
        ) or DEFAULT_DEV_USER_EMAIL
        resolved_user_id = (
            os.getenv("DEV_USER_ID", DEFAULT_DEV_USER_ID).strip()
            if dev_user_id is None
            else dev_user_id.strip()
        ) or DEFAULT_DEV_USER_ID
        return AuthenticatedUser(
            email=resolved_email,
            user_id=resolved_user_id,
            auth_source="local_dev",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
    )


def get_current_user(request: Request) -> AuthenticatedUser:
    """FastAPI dependency returning the authenticated doctor for the current request."""
    return authenticate_headers(request.headers)
