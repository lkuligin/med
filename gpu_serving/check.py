"""Ask an endpoint whether it is serving what we think, and answering usably.

    from gpu_serving.check import verify
    verify("http://127.0.0.1:8000", "google/gemma-4-26B-A4B-it")

This is not a warm-up - SGLang is warm before it answers at all. It catches the
failures that otherwise surface as bad results rather than as errors:

* the endpoint serves a different checkpoint than the one we meant to measure;
* the reply is truncated, which arrives as finish_reason="length" and an answer
  that looks merely short - no exception, no warning, and a parser downstream
  that quietly finds nothing;
* reasoning leaks into content because the wrong --reasoning-parser was passed,
  which makes every local answer structurally unlike the gateway's.

It takes a base URL rather than reading this package's state, so the same check
runs against the gateway - which is the point when the two are being compared.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass

PROBE = "Reply with exactly: PONG"
# Markers of a reasoning block that should have been split out of content.
LEAKED = re.compile(r"<(think|thinking|reasoning)>|<\|channel\|>", re.I)


@dataclass
class Result:
    ok: bool
    served_names: list[str]
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    problems: list[str]

    def report(self) -> str:
        lines = [f"models: {', '.join(self.served_names) or '(none)'}",
                 f"finish_reason: {self.finish_reason}",
                 f"tokens: {self.prompt_tokens} in / {self.completion_tokens} out",
                 f"content: {self.content[:200]!r}"]
        lines += [f"PROBLEM: {p}" for p in self.problems]
        lines.append("ok" if self.ok else "FAILED")
        return "\n".join(lines)


def _post(url: str, payload: dict, timeout: float = 120) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _get(url: str, timeout: float = 30) -> dict:
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer local"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def verify(base_url: str, expect_name: str, max_tokens: int = 256,
           extra_body: dict | None = None) -> Result:
    """Check the endpoint serves `expect_name` and returns a complete reply."""
    problems: list[str] = []

    listing = _get(f"{base_url}/v1/models")
    served = [entry["id"] for entry in listing.get("data", [])]
    if expect_name not in served:
        problems.append(
            f"expected {expect_name!r} among served models, got {served}")

    payload = {
        "model": expect_name,
        "messages": [{"role": "user", "content": PROBE}],
        "max_tokens": max_tokens,
        "temperature": 0,
        **(extra_body or {}),
    }
    reply = _post(f"{base_url}/v1/chat/completions", payload)
    choice = reply["choices"][0]
    content = choice["message"].get("content") or ""
    finish = choice.get("finish_reason", "")
    usage = reply.get("usage", {})

    if finish == "length":
        problems.append(
            f"reply hit the {max_tokens}-token limit; a real run would be "
            "silently truncated, so max_tokens for this model is too low")
    if not content.strip():
        problems.append("empty content - usually the whole budget went to reasoning")
    if LEAKED.search(content):
        problems.append("reasoning markers in content - wrong --reasoning-parser")

    return Result(
        ok=not problems,
        served_names=served,
        content=content,
        finish_reason=finish,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        problems=problems,
    )


__all__ = ["verify", "Result", "PROBE"]
