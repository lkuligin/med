"""Talk to the LLM gateway from Python, naming any model by a single string.

    from gdanschin_runtime.gateway import ask, list_models

    ask("gemini-3.8-flash", "Reply with exactly: PONG")
    ask("gpt-4.1-mini",     "...")
    ask("claude-opus-5",    "...")
    list_models("anthropic")          # what this token can actually reach

Open-weight models are reachable too, under short aliases:

    ask("gemma-4-26b", "...")         # google/gemma-4-26B-A4B-it, via sglang
    list_models("sglang")             # everything that provider serves

gemma-4-26b matters in particular: it is llm_monkeys' DEFAULT_MODEL, so step 2
can run the original experiment's model through the gateway, with no SGLang to
stand up and no Vertex credentials.

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
# The gateway also fronts internal providers (sglang, sglang-worker-slot* and
# others); they are deliberately absent, because the plan is to serve step 2
# from our own SGLang on the GPU box rather than through here.
# openrouter and yandex are left out too: their model endpoints did not answer
# in a usable shape, and claiming support without a passing call would be worse
# than a clear "unknown provider" error.
# provider -> (litellm prefix, path under the gateway, which gateway)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "openai": ("openai", "/proxy/openai", "external"),
    "anthropic": ("anthropic", "/proxy/anthropic", "external"),
    "google": ("gemini", "/proxy/google/v1beta", "external"),
    "deepseek": ("openai", "/proxy/deepseek", "external"),
    "xai": ("openai", "/proxy/xai", "external"),
    # Note the "/v1": path conventions differ per provider, and
    # /proxy/sglang/models 404s where /proxy/sglang/v1/models works.
    "sglang": ("openai", "/proxy/sglang/v1", "external"),
}

# On the internal gateway every open-weight model is its own provider, reached
# at /proxy/<provider>/v1 and OpenAI-compatible. Rather than freeze a list that
# will drift, any provider not named above is assumed to follow that shape.
# Known ones at the time of writing: deepseek-v4-flash-0731, gpt-oss-120b,
# qwen36-27b-fp8, qwen38-27b-fp8, glm5-fp8.
def _internal(provider: str) -> tuple[str, str, str]:
    return ("openai", f"/proxy/{provider}/v1", "internal")

# Short names for models whose real id is awkward. Two reasons they are needed
# rather than nice to have: the sglang ids contain a slash, which would other-
# wise be read as a provider prefix and route "google/gemma-..." to the Gemini
# proxy; and "gemma-4-26b" is the same alias llm_monkeys already uses for this
# model, so one name works in both places.
ALIASES: dict[str, str] = {
    # External gateway, via the sglang provider.
    "gemma-4-26b": "sglang/google/gemma-4-26B-A4B-it",
    "gemma-4-26b-a4b-it": "sglang/google/gemma-4-26B-A4B-it",
    "minimax-m2.7": "sglang/MiniMaxAI/MiniMax-M2.7",
    "qwen3-reranker-4b": "sglang/Qwen/Qwen3-Reranker-4B",
    # Internal gateway. Every id here was read from the provider itself rather
    # than copied from a list, since the wiki's tables have already gone stale
    # once. The gemma provider there reports its model as "unknown" and accepts
    # exactly that string - hence the odd-looking value.
    "deepseek-v4-flash": "deepseek-v4-flash-0731/deepseek-ai/DeepSeek-V4-Flash-0731",
    "glm5.3-flash": "glm5-fp8/zai-org/GLM-5.3-Flash",
    "gpt-oss-120b": "gpt-oss-120b/openai/gpt-oss-120b",
    "qwen3.6-27b": "qwen36-27b-fp8/Qwen/Qwen3.6-27B-FP8",
    "qwen3.8-27b": "qwen38-27b-fp8/Qwen/Qwen3.8-27B-FP8",
    "gemma-4-26b-internal": "gemma-4-26b-a4b-it/unknown",
    "qwen3.6-27b-noreasoning": "qwen36-27b-fp8/Qwen/Qwen3.6-27B-FP8",
    "qwen3.8-27b-noreasoning": "qwen38-27b-fp8/Qwen/Qwen3.8-27B-FP8",
}

# Extra request options some aliases carry. Qwen3 thinks by default and will
# spend the whole budget doing it: asked for medical facts it burned 4096
# tokens on reasoning and returned an empty string, with finish_reason=length
# and no error - which reads as "the model produced nothing" rather than "the
# limit was too low". Turning thinking off cuts that to ~1000 tokens.
#
# The switch has to be chat_template_kwargs. Putting "/no_think" in the prompt
# does nothing, because the gateway applies the chat template itself, and
# litellm rejects reasoning_effort for an openai-dialect provider.
#
# It is not a free speedup: without thinking the model produces about half as
# many facts, so -noreasoning is a different configuration rather than a faster
# version of the same one.
MODEL_EXTRAS: dict[str, dict] = {
    "qwen3.6-27b-noreasoning": {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "qwen3.8-27b-noreasoning": {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
}

# Model list endpoints, where they deviate from "<provider path>/models".
_MODEL_LIST_PATHS = {
    "google": "/proxy/google/v1beta/models",
    "anthropic": "/proxy/anthropic/v1/models",
    "sglang": "/proxy/sglang/v1/models",
}

# Matched in order against a bare model name.
_INFER = [
    (r"^(gpt|o\d|chatgpt|text-embedding|dall-e|whisper|tts|sora|codex|babbage|davinci)", "openai"),
    (r"^claude", "anthropic"),
    # Before the gemini rule: "gemma" and "gemini" share a prefix to the eye but
    # live on different providers, and gemma is not served by the Google proxy.
    (r"^gemma", "sglang"),
    (r"^gemini", "google"),
    (r"^deepseek", "deepseek"),
    (r"^grok", "xai"),
]


def _config_values() -> dict[str, str]:
    """Read LLM_GATEWAY_* settings from configs/gateway.conf."""
    values: dict[str, str] = {}
    if CONFIG.is_file():
        for line in CONFIG.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = re.match(r'\s*(LLM_GATEWAY_\w+)\s*=\s*[\'"]?(.*?)[\'"]?\s*$', line)
            if m:
                values[m.group(1)] = m.group(2)
    return values


def load_gateway(kind: str = "external") -> tuple[str, str]:
    """Return (url, token) for the "external" or "internal" gateway.

    They are separate deployments with separate tokens: the external one
    fronts the vendors, the internal one the open-weight models. Both
    addresses come from gateway.conf, which is neither committed nor mirrored.
    """
    if kind not in ("external", "internal"):
        raise ValueError(f"unknown gateway {kind!r}")
    suffix = "" if kind == "external" else "INTERNAL_"
    url_key, token_key = f"LLM_GATEWAY_{suffix}URL", f"LLM_GATEWAY_{suffix}TOKEN"

    url = os.getenv(url_key, "")
    token = os.getenv(token_key, "")
    if not url or not token:
        values = _config_values()
        url = url or values.get(url_key, "")
        token = token or values.get(token_key, "")

    if not url or not token or token == "paste-token-here":
        raise RuntimeError(
            f"the {kind} LLM gateway is not configured ({url_key}, {token_key}).\n"
            "  cp gdanschin_runtime/configs/gateway.conf.example "
            "gdanschin_runtime/configs/gateway.conf\n"
            f"  then paste the token. The two gateways use different ones."
        )
    return url.rstrip("/"), token


def resolve(model: str) -> tuple[str, str]:
    """Split a model string into (provider, bare model name).

    Accepts "provider/model" or a bare name whose provider is inferred.
    """
    model = ALIASES.get(model.lower(), model)

    if "/" in model:
        provider, _, bare = model.partition("/")
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
    prefix, path, kind = PROVIDERS.get(provider) or _internal(provider)
    url, token = load_gateway(kind)

    kwargs = {
        "model": f"{prefix}/{bare}",
        "api_base": f"{url}{path}",
        "api_key": token,
    }
    kwargs.update(MODEL_EXTRAS.get(model.lower(), {}))
    if provider == "google":
        # Without this the key would travel as ?key=, which the gateway ignores.
        kwargs["extra_headers"] = {"Authorization": f"Bearer {token}"}
    kwargs.update(overrides)
    return kwargs


def ask(model: str, prompt: str, max_tokens: int = 1024, **overrides):
    """Send one prompt to any gateway model and return the litellm response.

    Warns when the reply was cut off at max_tokens. That is the one failure
    here that looks like success: a reasoning model can spend the whole budget
    thinking and return an empty string, with no error anywhere, so the only
    signal is finish_reason.
    """
    import warnings

    import litellm

    response = litellm.completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        **completion_kwargs(model, **overrides),
    )

    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        details = getattr(response.usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None) or 0
        spent = f", {reasoning} of them on reasoning" if reasoning else ""
        warnings.warn(
            f"{model} hit max_tokens={max_tokens}{spent}; the reply is truncated "
            f"and may be empty. Raise max_tokens and re-run.",
            stacklevel=2,
        )
    return response


def list_models(provider: str) -> list[str]:
    """Ask the gateway which models of a provider this token can reach.

    Worth preferring over any written-down list: the wiki's Anthropic table is
    already stale, and asking is the only way to know what the token allows.
    """
    import json
    import urllib.request

    base, kind = (PROVIDERS.get(provider) or _internal(provider))[1:]
    url, token = load_gateway(kind)
    path = _MODEL_LIST_PATHS.get(provider, f"{base}/models")

    req = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)

    if provider == "google":
        return sorted(m["name"].replace("models/", "") for m in data.get("models", []))
    return sorted(m["id"] for m in data.get("data", []))


def list_providers(kind: str = "external") -> list[str]:
    """Return the provider names a gateway reports."""
    import json
    import urllib.request

    url, token = load_gateway(kind)
    req = urllib.request.Request(
        f"{url}/api/providers/list", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return sorted(p["name"] for p in json.load(resp).get("providers", []))


def catalogue(kind: str = "internal") -> dict[str, list[str]]:
    """Every provider on a gateway and the models it serves.

    On the internal gateway one provider is one model, so this is the way to
    see the open-weight catalogue; embedding and reranker providers appear here
    too and are simply not chat models.

    Providers that fail to answer are reported with an empty list rather than
    raising, since one sulking provider should not hide the rest.
    """
    out: dict[str, list[str]] = {}
    for provider in list_providers(kind):
        try:
            out[provider] = list_models(provider)
        except Exception:
            out[provider] = []
    return out


def aliases_for(model_id: str) -> list[str]:
    """Short names that resolve to a given "provider/model" string."""
    return sorted(k for k, v in ALIASES.items() if v == model_id)


__all__ = [
    "ask",
    "aliases_for",
    "catalogue",
    "MODEL_EXTRAS",
    "completion_kwargs",
    "list_models",
    "list_providers",
    "load_gateway",
    "resolve",
]
