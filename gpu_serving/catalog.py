"""The models this box can serve, and the topology each one is served with.

    from gpu_serving.catalog import SERVABLE, describe

    describe()                  # what can be served
    SERVABLE["gemma-4-26b"]     # one configuration

A name here identifies a *serving* configuration: the same weights with
thinking parsers wired differently, or at a different context length, is a
different entry, because it answers differently.

Nothing in this module knows what the models are used for. The only contract
this package offers the outside world is an OpenAI-compatible endpoint, so an
experiment runner needs a URL and nothing from this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Our four cards each hold 141 GB, and every checkpoint below fits on one of
# them - measured, not assumed. So the rule is: the smallest tp the weights
# allow, and the rest of the cards go to independent replicas.
#
# Tensor parallelism costs a cross-GPU synchronisation on every decode step and
# buys nothing once a model fits on one card; replicas scale nearly linearly
# with no communication at all. The tp2/tp4 in the reference playground scripts
# come from Kubernetes overlays tuned for a latency SLA on other hardware, and
# are not a claim that the weights need several cards.
CARD_MEMORY_GB = 141


@dataclass(frozen=True)
class ServedModel:
    """One servable configuration: which weights, and how they are launched."""

    name: str
    weights: str
    served_name: str
    weights_gb: float
    tp: int = 1
    context_length: int = 32768
    mem_fraction: float = 0.85
    sglang_args: tuple[str, ...] = ()
    note: str = ""

    def replicas(self, cards: int) -> int:
        """How many independent copies fit across `cards`, given this tp."""
        if cards % self.tp:
            raise ValueError(
                f"{self.name}: tp={self.tp} does not divide {cards} cards"
            )
        return cards // self.tp


SERVABLE: dict[str, ServedModel] = {
    "gemma-4-26b": ServedModel(
        name="gemma-4-26b",
        weights="google/gemma-4-26B-A4B-it",
        served_name="google/gemma-4-26B-A4B-it",
        weights_gb=49,
        context_length=132096,
        sglang_args=(
            "--chunked-prefill-size", "4096",
            "--kv-cache-dtype", "fp8_e4m3",
            "--constrained-json-disable-any-whitespace",
            "--page-size", "64",
            "--enable-metrics",
        ),
        note="the experiment's default model; the gateway serves it too, so it "
             "is the one configuration where local and gateway can be compared "
             "against results we already have. No --attention-backend here on "
             "purpose. SGLang rejects flashinfer for Gemma4 outright, and of "
             "the backends it does accept, trtllm_mha loads and serves but "
             "decodes garbage: measured 7/20 against triton's 13/20, every "
             "reply burning the full token budget on <pad>. The triton "
             "default is the right one, and is why the reference playground "
             "script names no backend either",
    ),
    "gpt-oss-120b": ServedModel(
        name="gpt-oss-120b",
        weights="openai/gpt-oss-120b",
        served_name="openai/gpt-oss-120b",
        weights_gb=61,
        mem_fraction=0.75,
        sglang_args=(
            "--trust-remote-code",
            "--reasoning-parser", "gpt-oss",
            "--tool-call-parser", "gpt-oss",
        ),
        note="120B total but MXFP4 with 4 of 128 experts active, so 61 GB on "
             "disk and fast to decode",
    ),
    "qwen3.8-27b": ServedModel(
        name="qwen3.8-27b",
        weights="Qwen/Qwen3.8-27B-FP8",
        served_name="Qwen/Qwen3.8-27B-FP8",
        weights_gb=28,
        mem_fraction=0.85,
        context_length=65536,
        sglang_args=(
            "--trust-remote-code",
            "--kv-cache-dtype", "fp8_e4m3",
            "--attention-backend", "flashinfer",
            "--chunked-prefill-size", "32768",
            "--max-prefill-tokens", "32768",
            "--reasoning-parser", "qwen3",
            "--tool-call-parser", "qwen3_coder",
            "--mamba-radix-cache-strategy", "extra_buffer",
            "--mamba-ssm-dtype", "float32",
        ),
        note="FP8 to match what the gateway serves; hybrid GDN, so the mamba "
             "flags are not optional. Weights are not on the box yet.",
    ),
}


def describe() -> None:
    """Print the catalogue."""
    for m in SERVABLE.values():
        print(f"  {m.name:16} {m.served_name:32} {m.weights_gb:>5.0f} GB  tp={m.tp}")
        if m.note:
            print(f"    {m.note}")


if __name__ == "__main__":
    describe()


__all__ = ["ServedModel", "SERVABLE", "describe", "CARD_MEMORY_GB"]
