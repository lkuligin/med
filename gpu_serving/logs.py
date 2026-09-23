"""Read what a server's log says about how hard it was actually worked.

    from gpu_serving.logs import summarise
    print(summarise(open("run/gemma-4-26b.log").read()).render())

Needs no GPU: a log copied back from the box summarises the same on a laptop.

SGLang prints one line per scheduler batch, with "label: value" fields. Which
labels appear depends on the version and on what is enabled - speculative
decoding adds acceptance lengths, disaggregation adds transfer counts - so the
parser takes every numeric field it finds rather than a fixed list, and the
summary reports the ones it recognises. A field that disappears in a future
release costs a line of output, not a crash.

The question this is here to answer is whether the four replicas were ever
busy. A run where #running-req stays low and #queue-req is always zero was
limited by the client, not by the GPUs, and its throughput number says nothing
about the hardware.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

# "label: 12.34" or "label: 45%" - labels may contain #, spaces, slashes and
# parenthesised units, and must not swallow the preceding comma.
FIELD = re.compile(r"([#A-Za-z][^,:]*?):\s*(-?\d+(?:\.\d+)?)\s*%?")
BATCH = re.compile(r"\b(Prefill|Decode) batch\b")
READY = re.compile(r"server is fired up|The server is (?:now )?ready", re.I)


@dataclass
class Batches:
    """Every numeric field seen, keyed by label, in the order encountered."""

    kind: str
    fields: dict[str, list[float]] = field(default_factory=dict)

    def add(self, label: str, value: float) -> None:
        self.fields.setdefault(label, []).append(value)

    @property
    def count(self) -> int:
        return max((len(v) for v in self.fields.values()), default=0)

    def get(self, label: str) -> list[float]:
        return self.fields.get(label, [])


@dataclass
class Summary:
    prefill: Batches
    decode: Batches
    ready: bool
    lines: int

    def render(self) -> str:
        out: list[str] = []
        out.append(f"{self.lines} log lines, "
                   f"{self.prefill.count} prefill / {self.decode.count} decode batches")
        if not self.ready:
            out.append("the log has no startup-complete line - the server may "
                       "never have finished loading")
        if not self.decode.count:
            out.append("no decode batches: nothing was generated from this log")
            return "\n".join(out)

        out.append("")
        for label, unit in (("gen throughput (token/s)", "tok/s"),
                            ("input throughput (token/s)", "tok/s")):
            values = self.decode.get(label) or self.prefill.get(label)
            if values:
                out.append(f"{label:34} median {_median(values):9.0f} {unit}"
                           f"   peak {max(values):9.0f} {unit}")

        running = self.decode.get("#running-req")
        if running:
            out.append(f"{'concurrent requests':34} median {_median(running):9.0f}"
                       f"        peak {max(running):9.0f}")
        queued = self.decode.get("#queue-req")
        if queued:
            waited = sum(1 for q in queued if q > 0) / len(queued)
            out.append(f"{'queued requests':34} peak   {max(queued):9.0f}"
                       f"        {waited:.0%} of batches had a queue")
        usage = self.decode.get("token usage")
        if usage:
            out.append(f"{'KV pool usage':34} peak   {max(usage):9.2f}")
        accept = self.decode.get("accept len")
        if accept:
            out.append(f"{'speculative accept len':34} mean   {_median(accept):9.2f}")
        hit = self.prefill.get("cache hit rate")
        if hit:
            out.append(f"{'prefix cache hit rate':34} mean   {_median(hit):9.1f} %")

        out.append("")
        out.append(self.verdict())
        return "\n".join(out)

    def verdict(self) -> str:
        """The one reading that changes what to do next."""
        running = self.decode.get("#running-req")
        queued = self.decode.get("#queue-req")
        if not running:
            return "no concurrency figures in this log."
        peak = max(running)
        never_queued = not queued or max(queued) == 0
        if peak <= 4 and never_queued:
            return (f"The server was never busy: at most {peak:.0f} requests in "
                    "flight and nothing ever queued. This measures the client, "
                    "not the GPUs - raise the run's concurrency.")
        if never_queued:
            return (f"Peak {peak:.0f} in flight, never queued: the server kept up, "
                    "so there is room to push harder.")
        return (f"Peak {peak:.0f} in flight with a queue forming: the server was "
                "the limit here, which is what a throughput run should look like.")


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def summarise(text: str) -> Summary:
    """Parse a server log. Tolerant of unknown and missing fields."""
    prefill, decode = Batches("prefill"), Batches("decode")
    ready = False
    lines = 0

    for line in text.splitlines():
        lines += 1
        if READY.search(line):
            ready = True
        match = BATCH.search(line)
        if not match:
            continue
        target = prefill if match.group(1) == "Prefill" else decode
        for label, value in FIELD.findall(line[match.end():]):
            target.add(label.strip(), float(value))

    return Summary(prefill=prefill, decode=decode, ready=ready, lines=lines)


__all__ = ["summarise", "Summary", "Batches"]
