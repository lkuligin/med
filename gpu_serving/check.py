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
  which makes every local answer structurally unlike the gateway's;
* the model generates degenerate text - padding tokens, or one token over and
  over - which an attention backend that SGLang accepts can still produce.

That last one is why there are two probes rather than one. A three-token
"reply PONG" passed a server whose real answers were streams of <pad>: the
short probe never generated enough for the fault to show. The second probe
asks for a few hundred tokens, which is what a real question costs.

It takes a base URL rather than reading this package's state, so the same check
runs against the gateway - which is the point when the two are being compared.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass

PROBE = "Reply with exactly: PONG"
# Long enough that decoding has to keep working for a few hundred tokens, and
# checkable without a grader: the reply either counts or it does not.
GENERATION_PROBE = (
    "List the cranial nerves in order, numbered I to XII, "
    "one per line, with a short note on what each one does."
)
# Markers of a reasoning block that should have been split out of content.
LEAKED = re.compile(r"<(think|thinking|reasoning)>|<\|channel\|>", re.I)
# Special tokens that should never reach content; <pad> runs are what a broken
# attention backend produced here.
SPECIAL = re.compile(r"<(pad|unk|s|/s|eos|bos)>", re.I)


@dataclass
class Result:
    ok: bool
    served_names: list[str]
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    problems: list[str]
    generated_tokens: int = 0
    generated_head: str = ""

    def report(self) -> str:
        lines = [f"models: {', '.join(self.served_names) or '(none)'}",
                 f"finish_reason: {self.finish_reason}",
                 f"tokens: {self.prompt_tokens} in / {self.completion_tokens} out",
                 f"content: {self.content[:200]!r}",
                 f"generation probe: {self.generated_tokens} tokens, "
                 f"{self.generated_head[:90]!r}"]
        lines += [f"PROBLEM: {p}" for p in self.problems]
        lines.append("ok" if self.ok else "FAILED")
        return "\n".join(lines)


def degenerate(text: str) -> str | None:
    """Say how the text is degenerate, or None if it looks like language.

    Catches the two shapes a broken decode takes: special tokens arriving as
    text, and one short string repeated until the budget runs out.
    """
    special = SPECIAL.search(text)
    if special:
        count = len(SPECIAL.findall(text))
        return f"{count} special token(s) in the text, first {special.group(0)!r}"

    words = text.split()
    if len(words) >= 40:
        commonest = max(set(words), key=words.count)
        share = words.count(commonest) / len(words)
        if share > 0.5:
            return f"{share:.0%} of the words are {commonest!r}"
    return None


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


def _ask(base_url, model, prompt, max_tokens, extra_body):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        **(extra_body or {}),
    }
    reply = _post(f"{base_url}/v1/chat/completions", payload)
    choice = reply["choices"][0]
    return (choice["message"].get("content") or "",
            choice.get("finish_reason", ""),
            reply.get("usage", {}))


def verify(base_url: str, expect_name: str, max_tokens: int = 256,
           extra_body: dict | None = None) -> Result:
    """Check the endpoint serves `expect_name` and generates usable text."""
    problems: list[str] = []

    listing = _get(f"{base_url}/v1/models")
    served = [entry["id"] for entry in listing.get("data", [])]
    if expect_name not in served:
        problems.append(
            f"expected {expect_name!r} among served models, got {served}")

    content, finish, usage = _ask(base_url, expect_name, PROBE, max_tokens, extra_body)

    if finish == "length":
        problems.append(
            f"reply hit the {max_tokens}-token limit; a real run would be "
            "silently truncated, so max_tokens for this model is too low")
    if not content.strip():
        problems.append("empty content - usually the whole budget went to reasoning")
    if LEAKED.search(content):
        problems.append("reasoning markers in content - wrong --reasoning-parser")

    # The probe that a three-token reply cannot fail.
    long_content, long_finish, long_usage = _ask(
        base_url, expect_name, GENERATION_PROBE, 400, extra_body)
    wrong = degenerate(long_content)
    if wrong:
        problems.append(
            f"degenerate output over {long_usage.get('completion_tokens', 0)} "
            f"tokens: {wrong}. The model is loaded but decoding wrongly - "
            f"suspect the attention backend.")
    if long_finish == "length":
        problems.append(
            "the 400-token probe did not finish on its own; the model may be "
            "running away rather than answering")

    return Result(
        ok=not problems,
        served_names=served,
        content=content,
        finish_reason=finish,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        problems=problems,
        generated_tokens=long_usage.get("completion_tokens", 0),
        generated_head=" ".join(long_content.split())[:120],
    )


__all__ = ["verify", "Result", "PROBE"]
