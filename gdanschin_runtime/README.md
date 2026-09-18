# gdanschin_runtime

Model-invocation layer (external LLMs and internal models served with SGLang)
for experiments built on top of `llm_monkeys`. It exists so that
provider-specific code does not leak into `llm_monkeys`, which we keep as a
frozen reference implementation.

## Boundaries

- Imports flow one way only: `gdanschin_runtime` -> `llm_monkeys`. Never back.
- Nothing about SGLang, specific providers, or their dependencies belongs in
  `llm_monkeys`.
- Only two kinds of change to `llm_monkeys` are allowed:
  1. fixing what is objectively broken (see "Known breakage");
  2. adding a hook whose default behaviour is byte-identical to the original.
  Each goes in its own commit, so the delta from the reference stays readable
  with a single `git log`.
- Invariant: with nothing configured, `llm_monkeys` behaves exactly as the
  author left it - through Vertex AI. Our mode is opt-in.
- Regression guard: `pytest` inside `llm_monkeys` must keep passing.

## Layout

    providers/      model clients (SGLang, external APIs) - our code
    adapters/       ADK BaseLlm wrappers over providers
    cli/            entrypoints for steps 1-3 that wire workflows together
    configs/        run configurations
    _bootstrap.py   puts llm_monkeys on sys.path (the only place that does)

## Two ways to plug in

### 1. Agent injection (per-agent, no env vars)

All three workflows accept ready-made agents and runners, falling back to the
LiteLlm factory only when nothing is passed:

- `OneShotInferenceWorkflow(config, agent=..., runner=...)`               - one_shot/workflow.py:335
- `CandidateInferenceWorkflow(config, fact_agent=..., answer_agent=...)`  - inference/workflow.py:109
- `VerifierWorkflow(config, verifier_agent=..., verifier_runner=...)`     - verifier/workflow.py:40

Injection is per-agent, so step 2 can run on our SGLang while the step 3 judge
stays on Vertex AI, in one process, with no global switch. Use this when the
fact agent and the answer agent need different models.

### 2. The MEDQA_MODEL_FACTORY hook (works with the stock CLIs)

`llm_monkeys/model_factory.py` is the single construction point for the model
behind every agent. By default it returns the stock `LiteLlm` adapter. Setting

    export MEDQA_MODEL_FACTORY="gdanschin_runtime.adapters.factory:build"

redirects construction to that callable, which receives the active config and
returns an ADK `BaseLlm`. This makes the stock `cli.py`, `inference/cli.py` and
`verifier/cli.py` route to our models without any injection code.

The factory is keyed on the config object, so it can branch on
`isinstance(cfg, VerifierConfig)` to keep the judge on Vertex AI while sending
candidate inference elsewhere. It cannot tell the fact agent from the answer
agent, since both share one `CandidateInferenceConfig` - use injection for that.

A malformed or unimportable spec raises instead of falling back. That is
deliberate: a silent fallback would route traffic and cost to Vertex AI
unnoticed.

## Model adapter contract

Plug in at the level of an ADK `BaseLlm` subclass (what `LiteLlm` is). Then
everything above the model - Runner, sessions, retry with exponential backoff,
token accounting, incremental saving, parsers - is reused as is.

Requirements that follow from the code:

1. Consumption goes through `runner.run_async()` -> ADK events. Text is read
   from `event.content.parts[].text`; parts with `part.thought=True` go to a
   separate buffer (inference/workflow.py:186).
2. `event.usage_metadata` must expose `prompt_token_count`,
   `candidates_token_count`, `total_token_count`, `cached_content_token_count`
   and `thoughts_token_count`. Otherwise `StepTokenUsage.from_usage_metadata`
   (inference/_schemas.py:41) silently returns zeros and all token statistics
   die **without raising**. This is the easiest part of the contract to miss.
3. The fact agent is built with `output_schema=MedicalFacts` and
   `response_mime_type="application/json"` - the adapter should forward this to
   SGLang guided decoding. The fact parser has fallbacks (schema -> fenced JSON
   -> regex -> bullets), so degradation is graceful, but fact quality drops.

## Known breakage in llm_monkeys (fix before running)

- `inference/_dataset.py` is **missing** - not on disk, not in git history.
  `inference/workflow.py:23` imports `format_fact_generation_prompt`,
  `format_answer_generation_prompt` and `load_difficult_questions` from it.
  Step 2 does not import at all; `tests/unit/inference/` cannot be collected.
- `resolve_model_name` (config.py:70) force-rewrites anything starting with
  `openai/gpt-oss` or `google/gemma` to `vertex_ai/...`. Locally served models
  with those names will **silently go to Vertex AI**.
- The `llm_monkeys` README points at a non-existent `run_one_shot_inference.py`;
  the real entrypoint is `cli.py`.

## Data status

- `llm_monkeys/difficult_questions.csv` exists: 483 difficult questions, ids
  0..1272, i.e. the full MedQA test split (1273). Enough to start step 2.
- The raw `results_*.json` from step 1 are gone. They are only needed for
  `--step1-file` in `analyze_verifier.py` (comparison against the one-shot
  baseline).
- What produced the current CSV is recorded nowhere (AGENTS.md says "Gemma 3,
  Gemma 4", the README example shows gemma4/gemini/gpt-oss-20b, and Gemma 3 is
  absent from `SUPPORTED_MODELS`). The exact model set is not reproducible.
- `question_id` is the row index within the split, so the CSV is only valid for
  the same dataset/config/split. A version bump on HF would silently shift the
  mapping.

## Judge (step 3)

Must stay as is: `vertex_ai/gemini-3.8-flash` (config.py:18), `temperature=0.0`,
`max_tokens=512`, structured output `FactVerification`. Vertex AI access is
therefore required, but the gcloud CLI is not - a service-account JSON key via
`GOOGLE_APPLICATION_CREDENTIALS` is enough.

## Syncing to the GPU box

`sync/` mirrors this repository to `gpu.example.com:/home/gdanschin/Projects/med`,
one way only.

    ./gdanschin_runtime/sync/sync.sh --dry-run   # show what would be sent
    ./gdanschin_runtime/sync/sync.sh --once      # sync once and exit
    ./gdanschin_runtime/sync/sync.sh             # watch and sync until Ctrl-C

Watching needs fswatch: `brew install fswatch`.

lsyncd was the original plan and `sync/lsyncd-med.lua` is still here, but its
macOS backend opens `/dev/fsevents` directly, which is root-only, so it fails
with `Cannot access /dev/fsevents monitor! (1:Operation not permitted)` unless
run under sudo. fswatch uses the public FSEvents API and needs no privileges.
The lua config remains valid for a Linux host, and shares the same exclude
list.

The sync is a true mirror: `--delete` removes remote files that no longer
exist locally, so a module deleted here cannot linger there and get imported.

Remote-generated data is protected by `sync/rsync-exclude.txt` alone. Excluded
paths are neither transferred nor deleted - that is standard rsync behaviour,
since excluded receiver files are shielded from `--delete` (only
`--delete-excluded` would remove them, and we never pass it).

**On the GPU box, write only into these directories.** Anything created
outside them does not exist locally and WILL be deleted on the next sync:

| Directory  | For |
|---|---|
| `data/`    | datasets and downloaded inputs |
| `results/` | experiment outputs: json, csv, plots |
| `cache/`   | model and dataset caches - point `HF_HOME` here |
| `logs/`    | run logs |
| `scratch/` | ad-hoc files made on the remote: debug scripts, notes, patched configs |
| `.venv/`   | the remote's own environment |

Runtime leftovers a run drops outside those directories are excluded too, so
they survive as well: `__pycache__/`, `*.pyc`, `.pytest_cache/`,
`.ipynb_checkpoints/`, `wandb/`, `nohup.out`, `.coverage`, `*.tmp`, `*.swp`.

Stale bytecode is not a concern: CPython compares the exact mtime recorded in
a `.pyc` against its source, and `rsync -a` preserves mtimes, so any mismatch
forces a recompile - including after checking out an older revision locally.

**Stop the watcher before a long run.** The sync is live, so editing a file
here while a multi-hour job runs there changes its code mid-flight. Modules
already imported keep their loaded version, but anything imported later picks
up the new code, and the results silently mix two revisions. Push once with
`--once` and leave the watcher off for the duration.

They are unanchored patterns, so `llm_monkeys/results/` is covered as well as
a top-level `results/`. As a safety net, the default output filenames the code
writes into its working directory (`results_*.json`, `simple_questions.csv`,
`*.png`, `*.log`) are excluded too, so a run left on default paths is still
protected.

`rsync-exclude.txt` is the single source of truth, shared by the lsyncd config
and by `--dry-run`. Two things to keep in mind when editing it:

- `difficult_questions.csv` must stay included. It is tracked in git and is the
  input to step 2, so a blanket `*.csv` exclude would break the remote run.
  Because it is synced, anything written to it on the remote would be reverted
  by the next mirror pass, so `separate_result.py` now refuses to write it and
  emits `data/difficult_questions.candidate.csv` instead. `data/` is both
  gitignored and excluded here, so the candidate cannot be committed or shipped
  by accident; promote it once reviewed:

      ./gdanschin_runtime/promote_difficult_questions.sh

  The script copies the candidate over the tracked list and reports the
  question count before and after. Staging and committing stay manual.
- Output patterns are deliberately unanchored (`results/`, not `/results/`),
  because runs write relative to their working directory, which is usually
  `llm_monkeys/` rather than the repository root.

The legacy sibling directories (`med/`, `clustering/`, `clusters/`,
`simulations/`) are excluded; `med/` alone is 16 MB of the 21 MB repository and
is not needed to run llm_monkeys. Uncomment them if that changes.

Host, user and remote path are set at the top of `sync.sh`; the identity file
and source root can be overridden with `MED_SYNC_IDENTITY` and
`MED_SYNC_SOURCE`.

## Opt-ins

Everything we change in `llm_monkeys` is off by default, so an unconfigured
checkout behaves exactly as the original authors left it. `env.sh` collects our
opt-ins in one place:

    source gdanschin_runtime/env.sh

- `MEDQA_DIFFICULT_CANDIDATE` - send difficult-question output to a candidate
  file and refuse writes to the tracked `difficult_questions.csv`. Unset, the
  upstream behaviour applies: `--difficult-output` writes nothing unless asked,
  and may target the tracked file directly.
- `MEDQA_MODEL_FACTORY` - build models with our own client instead of the stock
  `LiteLlm` adapter. Left commented out until `adapters/factory.py` exists,
  since an unimportable spec fails loudly on purpose.
