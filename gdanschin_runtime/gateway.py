"""Talk to the LLM gateway from Python, naming any model by a single string.

    from gdanschin_runtime.gateway import ask, list_models

    ask("gemini-3.8-flash", "Reply with exactly: PONG")
    ask("gpt-4.1-mini",     "...")
    ask("claude-opus-5",    "...")
    list_models("anthropic")          # what this token can actually reach

External vendor models only. Internal providers the gateway also fronts are
left out on purpose - see PROVIDERS.

The provider is inferred from the model name and can be given explicitly as
"provider/model" when a name is ambiguous or new.

Three gateway details are handled here so they do not have to be rediscovered:

1. The Google proxy needs "/v1beta" appended to api_base. LiteLLM adds
   "/models/<model>:generateContent" itself, so without it every call 404s with
   an empty GeminiException that names no cause.
2. The gateway authenticates with an Authorization: Bearer header. LiteLLM's
   gemini provider otherwise sends the key as a ?key= query parameter, which the
   gateway ignores. The openai and anthropic providers use the header already.
3. Gemini 3.x spends most of its output budget on reasoning tokens - about 90 of
   them to answer "PONG". max_tokens must cover thinking plus the answer, or the
   reply is silently truncated or empty. Hence a roomy default.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "configs" / "gateway.conf"

# provider -> (litellm prefix, path under the gateway)
#
# External vendors only, and only ones verified to answer through the gateway.
# The gateway also fronts internal providers (sglang, sglang-worker-slot*,
# internal-llm, qwen3-reranker-4b); they are deliberately absent, because the plan is
# to serve step 2 from our own SGLang on the GPU box rather than through here.
# openrouter and yandex are left out too: their model endpoints did not answer
# in a usable shape, and claiming support without a passing call would be worse
# than a clear "unknown provider" error.
PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("openai", "/proxy/openai"),
    "anthropic": ("anthropic", "/proxy/anthropic"),
    "google": ("gemini", "/proxy/google/v1beta"),
    "deepseek": ("openai", "/proxy/deepseek"),
    "xai": ("openai", "/proxy/xai"),
}

# Matched in order against a bare model name.
_INFER = [
    (r"^(gpt|o\d|chatgpt|text-embedding|dall-e|whisper|tts|sora|codex|babbage|davinci)", "openai"),
    (r"^claude", "anthropic"),
    (r"^gemini", "google"),
    (r"^deepseek", "deepseek"),
    (r"^grok", "xai"),
]


def load_gateway() -> tuple[str, str]:
    """Return (url, token) from the environment, falling back to configs/gateway.conf."""
    url = os.getenv("LLM_GATEWAY_URL", "")
    token = os.getenv("LLM_GATEWAY_TOKEN", "")

    if (not url or not token) and CONFIG.is_file():
        for line in CONFIG.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = re.match(r'\s*(LLM_GATEWAY_\w+)\s*=\s*[\'"]?(.*?)[\'"]?\s*$', line)
            if m:
                key, value = m.groups()
                if key == "LLM_GATEWAY_URL" and not url:
                    url = value
                elif key == "LLM_GATEWAY_TOKEN" and not token:
                    token = value

    if not url or not token or token == "paste-token-here":
        raise RuntimeError(
            "LLM gateway is not configured.\n"
            "  cp gdanschin_runtime/configs/gateway.conf.example "
            "gdanschin_runtime/configs/gateway.conf\n"
            "  then paste the token, or set LLM_GATEWAY_URL / LLM_GATEWAY_TOKEN"
        )
    return url.rstrip("/"), token


def resolve(model: str) -> tuple[str, str]:
    """Split a model string into (provider, bare model name).

    Accepts "provider/model" or a bare name whose provider is inferred.
    """
    if "/" in model:
        provider, _, bare = model.partition("/")
        if provider not in PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}; known: {', '.join(sorted(PROVIDERS))}"
            )
        return provider, bare

    for pattern, provider in _INFER:
        if re.match(pattern, model, re.IGNORECASE):
            return provider, model

    raise ValueError(
        f"cannot tell which provider serves {model!r}. "
        f"Name it explicitly, e.g. 'openai/{model}'. "
        f"Known providers: {', '.join(sorted(PROVIDERS))}"
    )


def completion_kwargs(model: str, **overrides) -> dict:
    """Build the litellm.completion keyword arguments for a gateway model."""
    provider, bare = resolve(model)
    prefix, path = PROVIDERS[provider]
    url, token = load_gateway()

    kwargs = {
        "model": f"{prefix}/{bare}",
        "api_base": f"{url}{path}",
        "api_key": token,
    }
    if provider == "google":
        # Without this the key would travel as ?key=, which the gateway ignores.
        kwargs["extra_headers"] = {"Authorization": f"Bearer {token}"}
    kwargs.update(overrides)
    return kwargs


def ask(model: str, prompt: str, max_tokens: int = 1024, **overrides):
    """Send one prompt to any gateway model and return the litellm response."""
    import litellm

    return litellm.completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        **completion_kwargs(model, **overrides),
    )


def list_models(provider: str) -> list[str]:
    """Ask the gateway which models of a provider this token can reach.

    Worth preferring over any written-down list: the wiki's Anthropic table is
    already stale, and asking is the only way to know what the token allows.
    """
    import json
    import urllib.request

    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}")
    url, token = load_gateway()
    path = {
        "google": "/proxy/google/v1beta/models",
        "anthropic": "/proxy/anthropic/v1/models",
    }.get(provider, f"{PROVIDERS[provider][1]}/models")

    req = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)

    if provider == "google":
        return sorted(m["name"].replace("models/", "") for m in data.get("models", []))
    return sorted(m["id"] for m in data.get("data", []))


def list_providers() -> list[str]:
    """Return the provider names the gateway itself reports."""
    import json
    import urllib.request

    url, token = load_gateway()
    req = urllib.request.Request(
        f"{url}/api/providers/list", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return sorted(p["name"] for p in json.load(resp).get("providers", []))


__all__ = ["ask", "completion_kwargs", "list_models", "list_providers", "load_gateway", "resolve"]
