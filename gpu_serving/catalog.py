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


# Our four cards each hold 141 GB. The rule is the smallest tp the weights
# allow - measured, not assumed - and the rest of the cards go to independent
# replicas.
#
# Tensor parallelism costs a cross-GPU synchronisation on every decode step and
# buys nothing once a model fits on one card; replicas scale nearly linearly
# with no communication at all. The tp2/tp4 in the reference playground scripts
# come from Kubernetes overlays tuned for a latency SLA on other hardware, and
# are not a claim that the weights need several cards.
#
# Everything up to Qwen3.8-27B fits one card and is served tp=1. The judge
# candidates below do not, and their tp is arithmetic: weights_gb / tp has to
# leave the static budget room for a KV pool on every card.
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
    # Sent with every probe request. Some models answer entirely inside a
    # thinking block unless told not to, which leaves content empty and makes
    # a verification that omits this test a configuration nobody runs.
    request_extras: dict[str, Any] = field(default_factory=dict)
    # Set when this SGLang cannot serve the weights at all. The entry stays,
    # because the topology and flags are worked out and the day the support
    # lands only this line has to go; but a launch has to refuse now rather
    # than fail deep inside a load nobody is watching.
    unsupported: str = ""
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
    "gemma-4-e2b": ServedModel(
        name="gemma-4-e2b",
        weights="google/gemma-4-E2B-it",
        served_name="google/gemma-4-E2B-it",
        weights_gb=9.6,
        sglang_args=(
            "--reasoning-parser", "gemma4",
            "--tool-call-parser", "gemma4",
        ),
        note="the smallest Gemma 4; flags from the reference playground's "
             "e2b.sh, which unlike the 26B script does name the parsers",
    ),
    "gpt-oss-20b": ServedModel(
        name="gpt-oss-20b",
        weights="openai/gpt-oss-20b",
        served_name="openai/gpt-oss-20b",
        weights_gb=13,
        mem_fraction=0.75,
        sglang_args=(
            "--trust-remote-code",
            "--reasoning-parser", "gpt-oss",
            "--tool-call-parser", "gpt-oss",
        ),
        note="the smaller of the two GPT-OSS models; there is no third",
    ),
    "qwen3.6-27b": ServedModel(
        name="qwen3.6-27b",
        weights="Qwen/Qwen3.6-27B-FP8",
        served_name="Qwen/Qwen3.6-27B-FP8",
        weights_gb=28,
        context_length=65536,
        sglang_args=(
            "--trust-remote-code",
            "--linear-attn-backend", "triton",
            "--mamba-ssm-dtype", "bfloat16",
            "--reasoning-parser", "qwen3",
            "--tool-call-parser", "qwen3_coder",
        ),
        request_extras={"chat_template_kwargs": {"enable_thinking": False}},
        note="the FP8 the gateway serves, and the exact size twin of "
             "Qwen3.8-27B-FP8 - same family one generation apart, which is "
             "the cleanest generation-over-generation comparison this box "
             "can make. Flags from the production overlay qwen36-27b-*.yaml, "
             "which unlike the 3.8 recipe names the linear-attention backend.",
    ),
    "qwen3.6-35b-a3b": ServedModel(
        name="qwen3.6-35b-a3b",
        weights="Qwen/Qwen3.6-35B-A3B-FP8",
        served_name="Qwen/Qwen3.6-35B-A3B-FP8",
        weights_gb=35,
        context_length=65536,
        sglang_args=(
            "--trust-remote-code",
            "--kv-cache-dtype", "fp8_e4m3",
            "--attention-backend", "flashinfer",
            "--page-size", "64",
            "--reasoning-parser", "qwen3",
            "--tool-call-parser", "qwen3_coder",
        ),
        request_extras={"chat_template_kwargs": {"enable_thinking": False}},
        note="already on the box and never served. The reference playground "
             "script for it also sets EAGLE speculative decoding, which is "
             "left out here: that is a throughput tuning for a production "
             "SLA, and it has never been run on this hardware.",
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
        # How the gateway serves it (-noreasoning), and therefore how we run
        # it. Unlike 3.5 this model does answer into content by default, so
        # the probe passed either way - which is exactly why a probe should
        # test the configuration in use rather than a convenient one.
        request_extras={"chat_template_kwargs": {"enable_thinking": False}},
        note="FP8 to match what the gateway serves; hybrid GDN, so the mamba "
             "flags are not optional. Weights are not on the box yet.",
    ),
}


# The three checkpoints big enough to be worth judging with. A judge answers a
# 500-token prompt with a one-bit verdict tens of thousands of times over, so
# what it needs is knowledge and throughput, not context: every one of these
# is capped far below the context its config allows, because each token of the
# cap is KV pool taken from concurrency.
SERVABLE.update({
    "qwen3.5-122b-a10b": ServedModel(
        name="qwen3.5-122b-a10b",
        weights="Qwen/Qwen3.5-122B-A10B-FP8",
        served_name="Qwen/Qwen3.5-122B-A10B-FP8",
        weights_gb=118,
        tp=2,
        context_length=32768,
        sglang_args=(
            "--trust-remote-code",
            "--linear-attn-backend", "triton",
            "--mamba-ssm-dtype", "float32",
            "--reasoning-parser", "qwen3",
        ),
        request_extras={"chat_template_kwargs": {"enable_thinking": False}},
        note="the most knowledge that fits on this box: 122B total but 8 of "
             "256 experts active, so it decodes like a 10B. tp=2 because 118 "
             "GB does not fit a 141 GB card with a KV pool behind it; at tp=2 "
             "each card holds 59 GB and keeps ~60 GB of pool, and the four "
             "cards run two independent replicas. Hybrid attention - "
             "layer_types alternates linear with full every fourth layer - so "
             "the mamba flags are not optional, and float32 is what its own "
             "config asks for. MTP draft layers ship with the weights and are "
             "left unused here, like EAGLE on 3.6-35B-A3B: untried on this "
             "hardware. The checkpoint is a vision model; we serve the text "
             "side and never send it an image.",
    ),
    "minimax-m2.5": ServedModel(
        name="minimax-m2.5",
        weights="MiniMaxAI/MiniMax-M2.5",
        served_name="MiniMaxAI/MiniMax-M2.5",
        weights_gb=214,
        tp=2,
        context_length=32768,
        mem_fraction=0.9,
        sglang_args=(
            "--trust-remote-code",
            "--reasoning-parser", "minimax-append-think",
        ),
        note="plain MoE, full attention throughout, so none of the hybrid "
             "flags apply. tp=2 puts 107 GB on each card and the 0.9 budget "
             "leaves about 19 GB of pool - thin, but a judge's prompts are "
             "500 tokens and it is concurrency that matters, not depth. If it "
             "runs out of memory loading, tp=4 is the fallback: half the "
             "throughput, four times the pool. The reference overlay for "
             "M2.7 also sets --ep-size equal to tp; left out until measured. "
             "A thinking model, hence the parser - whether the thinking can "
             "be turned off for judging is one of the things to find out.",
    ),
    "glm-5.3-flash": ServedModel(
        name="glm-5.3-flash",
        weights="zai-org/GLM-5.3-Flash",
        served_name="zai-org/GLM-5.3-Flash",
        weights_gb=306,
        tp=4,
        context_length=32768,
        mem_fraction=0.88,
        sglang_args=(
            "--trust-remote-code",
            "--reasoning-parser", "glm45",
        ),
        unsupported="SGLang 0.5.19 has no Glm5NextForConditionalGeneration in "
                    "its model registry - it carries Glm4Moe* and GlmMoeDsa*, "
                    "and --trust-remote-code does not add an implementation. "
                    "It needs a newer SGLang, in a venv of its own: upgrading "
                    "the one the sweep switches models with would strand the "
                    "sweep.",
        note="306 GB leaves no choice but tp=4, which means a single replica "
             "on all four cards - the worst topology there is for a judge "
             "making hundreds of thousands of short calls, whatever the "
             "model's quality. Kept for the day SGLang supports it. Its "
             "bigger sibling GLM-5.2-FP8 is on the box too at 704 GB and is "
             "not an entry at all: four cards hold 564.",
    ),
})


# Qwen ships nothing under 27B in 3.6 or 3.8, so small Qwen means dropping
# back to 3.5. These four are one family served one way, which is why they are
# generated rather than written out four times.
#
# No reference script exists for any of them. The reasoning parser is the
# family's and is UNVERIFIED here; no tool-call parser is set, because
# guessing one is worse than having none. The generation probe in check.py is
# what will say whether reasoning leaks into content.
QWEN35_SMALL_GB = {"0.8B": 2, "2B": 4, "4B": 10, "9B": 20}


def _qwen35_small(label: str, weights_gb: float) -> ServedModel:
    repo = f"Qwen/Qwen3.5-{label}"
    return ServedModel(
        name=f"qwen3.5-{label.lower()}",
        weights=repo,
        served_name=repo,
        weights_gb=weights_gb,
        sglang_args=("--trust-remote-code", "--reasoning-parser", "qwen3"),
        # Measured on 0.8B: without this the whole answer arrives in
        # reasoning_content and content is empty, so every reply looks blank.
        request_extras={"chat_template_kwargs": {"enable_thinking": False}},
        note="Qwen3.5 answers inside a thinking block unless told not to",
    )


SERVABLE.update({
    model.name: model
    for model in (_qwen35_small(label, gb) for label, gb in QWEN35_SMALL_GB.items())
})


def describe() -> None:
    """Print the catalogue."""
    for m in SERVABLE.values():
        print(f"  {m.name:18} {m.served_name:32} {m.weights_gb:>5.0f} GB  tp={m.tp}")
        if m.unsupported:
            print(f"    CANNOT BE SERVED: {m.unsupported}")
        if m.note:
            print(f"    {m.note}")


if __name__ == "__main__":
    describe()


__all__ = ["ServedModel", "SERVABLE", "describe", "CARD_MEMORY_GB"]
