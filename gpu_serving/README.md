# gpu_serving

Serve our own models on the GPU box, one at a time, for experiment runs.

This package is self-contained on purpose. It imports nothing from
`llm_monkeys` or `gdanschin_runtime`, and they import nothing from it: the only
contract it offers is *an OpenAI-compatible endpoint on a known port*. Its
environment, its pins and its runtime state all live inside the directory. A runner
that wants to use a locally served model needs a URL and the served name, which
it can read from `/v1/models` — no shared constants, nothing to drift. Moving
this directory into its own repository is a `git mv`.

It runs on any Python from 3.12, and is standard library only. The SGLang virtualenv is what it *launches*,
not what it needs, so `serve.sh status` keeps working while that venv is being
installed, and the package runs under any Python on any box. In practice that
means `serve.sh` runs under the *system* interpreter — 3.12 on the GPU box,
3.14 on a laptop — while the environment it launches is always 3.12. Keep the
code inside that range; nothing here should need a newer feature.

## Use

```bash
./gpu_serving/serve.sh list                 # what can be served
./gpu_serving/serve.sh command gemma-4-26b  # print the launch command, run nothing
./gpu_serving/serve.sh up gemma-4-26b       # start, wait until it can generate, verify
./gpu_serving/serve.sh status
./gpu_serving/serve.sh down                 # stop, wait for the cards to be released
./gpu_serving/serve.sh fetch qwen3.8-27b    # download weights (tens of GB)
./gpu_serving/serve.sh stats [log]          # how hard the server was worked
```

`list`, `command` and `stats` work anywhere. `up` needs the cards, the driver
and the environment, and on a machine without them it prints what is missing
and exits 1 rather than failing somewhere deep. `status` says the same.

`up` returns only once the server can generate, and `down` only once the driver
reports the cards free, so a sweep can put them back to back without sleeps.

## Setup

```bash
cp gpu_serving/configs/serving.conf.example gpu_serving/configs/serving.conf
```

Then build the serving environment, on the box (the wheels are CUDA builds, so
there is nothing to install on a laptop):

```bash
./gpu_serving/setup_env.sh
```

It creates `gpu_serving/.venv` from `requirements.txt`, with the prompt name
`serving_env` — the pipeline's venv calls itself `monkeys_env`, and on the box
the two are a tmux window apart, so the prompt says which shell can launch a
server. Override with `GPU_SERVING_ENV_NAME`. That environment is
separate from the pipeline's because SGLang pins its own torch/CUDA stack, and
it lives beside this package so that moving the directory moves everything it
needs. `.venv` at any depth is already ignored by git and by the mirror, so
each machine builds its own.

## Design notes

**Data parallel, not tensor parallel.** Every checkpoint we serve fits on one
141 GB card, so the topology is `--tp-size 1 --dp-size 4`: four independent
replicas behind one cache-aware router, one endpoint, no cross-GPU traffic. The
`tp2`/`tp4` in the reference playground scripts come from Kubernetes overlays
tuned for a latency SLA on other hardware. `tests/` asserts the assumption that
every catalogue entry still fits one card.

**Readiness is not liveness.** SGLang captures CUDA graphs during startup and
is warm by the time it answers, so nothing warms it up here. But `/health`
answers before the model can generate; only `/health_generate` waits for that.

**Stopping is not done when the process exits.** With replicas there are
several workers, and a card is free only when the driver says so. `down` polls
`nvidia-smi` before returning, because starting the next model too early fails
with an allocation error that names no cause.

**Verification catches silent wrongness, not crashes.** `check.verify` asserts
the endpoint serves the checkpoint we meant, that the reply did not stop on
`finish_reason="length"`, and that no reasoning leaked into `content`. Each of
those otherwise arrives as bad results rather than as an error. It takes a base
URL, so the same check runs against the gateway when the two are compared.

**A log is evidence about the client as much as the server.** `stats` reads a
server log — on any machine, so one copied back from the box summarises on a
laptop — and ends with the reading that changes what to do next: if
`#running-req` stayed low and nothing ever queued, the run was limited by the
runner's concurrency and its throughput figure says nothing about the GPUs.
Fields are parsed generically, because which ones SGLang prints depends on the
version and on what is enabled.

**Weights come from two places, and only one of them is ours.**
`/mnt/data/models` holds flat `<org>/<name>` directories, belongs to another
user, and is read-only for us — so nothing new can go there. Anything we fetch
lands in the Hugging Face cache at `GPU_SERVING_HF_HOME`, and a model that is
not in any flat root is handed to SGLang as its repo id, which `HF_HOME`
resolves. That keeps cache snapshot paths out of the catalogue. Serving runs
with `HF_HUB_OFFLINE=1`: starting a server is not the moment to find out a
download is needed, and `fetch` is where that belongs.

**Compiled kernels are cached, deliberately outside this directory.**
`GPU_SERVING_KERNEL_CACHE` (default `~/.cache/sglang`) is handed to SGLang,
its JIT layer and DeepGEMM, all three of which read a different variable. An
FP8 checkpoint spends minutes at startup warming DeepGEMM — Qwen3.8-27B-FP8
took far longer to become ready than the bf16 and MXFP4 models — and that is
a one-off cost only while the cache survives. Under the repository the mirror
would delete it between runs and the wait would return every time.

**Cards 4–7 only.** Cards 0–3 on this box belong to other people. The
allocation lives in `configs/serving.conf`, not in a flag someone can pass.
