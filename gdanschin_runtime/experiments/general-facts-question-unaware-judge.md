# general-facts-question-unaware-judge

The name of this pair of prompts. Step 2 writes general, self-contained
medical facts instead of case-specific ones; step 3 judges each fact with no
access to the question or the options.
The aim is facts that can be confirmed or refuted on their own, and a judge
that cannot answer the question for the candidate.

## What it changes

| Step | Switch | Value | Reference value |
|---|---|---|---|
| 2, fact generation | `--fact-prompt` | `background` | `reference` |
| 3, verification | `--judge-prompt` | `fact-only` | `reference` |

Both prompts live beside the reference ones and are selected by the flags
above; the defaults are the authors' prompts, unchanged.

- `llm_monkeys/inference/_prompts.py`, `FACT_PROMPTS["background"]`: the
  reference system instruction without "relevant to answering the question
  accurately"; the task asks for atomic, self-contained, falsifiable facts about
  the conditions, drugs and mechanisms the question and options mention, never
  about the patient or an option, two or three per option, at most 30.
- `llm_monkeys/verifier/_prompts.py`, `JUDGE_PROMPTS["fact-only"]`: the judge
  sees one statement and nothing else; 0 if any part is wrong, if it misstates
  frequency or strength, or if it refers to a patient or case that is not given.

## Runs

All on the box, under
`llm_monkeys/results/experiments/general-facts-question-unaware-judge/`, which
keeps the store's own layout (`med_qa/facts-pipeline/<run>/...`), so any tool
reads it with `--results-dir` pointed there. A copy of this file sits there as
`README.md`. The run summaries record `fact_prompt` and `judge_prompt`.

| Run | Judge dir | Questions | k | Notes |
|---|---|---|---|---|
| `gemma-4-26b-local-background-2` | `glm-5.3-flash-local-fact-only` | 483 (difficult_questions.csv) | 50 | the result below |
| `gemma-4-26b-background` | `glm-5.3-flash-local-fact-only`, `glm5.3-flash-fact-only` (8 verdicts, stopped) | 30 (`data/fact_prompt_probe_30.csv`) | 20 | pilot via the gateway, max_tokens 1024 |
| `gemma-4-26b-local-background` | none | partial | 50 | invalid: served with `--constrained-json-disable-any-whitespace`, 65% empty fact lists |

Reference for comparison: run `gemma-4-26b` in the main store,
`llm_monkeys/results/med_qa/facts-pipeline/` (gateway, reference prompts,
max_tokens 1024, k=50) with judges `glm-5.3-flash-local`, `gemini-3.8-flash`,
`deepseek-v4-flash-think-high`.

## Reproduce

Serve gemma-4-26b locally (the catalogue entry no longer passes
`--constrained-json-disable-any-whitespace`), then from `llm_monkeys/` with
`gdanschin_runtime/env.sh` sourced:

```bash
python -m inference.cli --results-dir results/experiments/general-facts-question-unaware-judge --model google/gemma-4-26B-A4B-it --run-name gemma-4-26b-local-background-2 --fact-prompt background --difficult-questions difficult_questions.csv --dataset bigbio/med_qa --n-candidates 50 --concurrency 50 --max-tokens 4096
```

The workflow takes questions one at a time, so the run used four disjoint
shards of the list in parallel (`data/local_background_chain.sh` on the box).
Then serve glm-5.3-flash and:

```bash
python -m verifier.cli --results-dir results/experiments/general-facts-question-unaware-judge --run-name gemma-4-26b-local-background-2 --judge-name glm-5.3-flash-local-fact-only --judge-prompt fact-only --model zai-org/GLM-5.3-Flash --temperature 1.0 --max-tokens 16384 --request-timeout 900 --dataset bigbio/med_qa --difficult-questions difficult_questions.csv --concurrency 16
```

## Results (2026-09-24, 483 difficult MedQA questions, k=50)

Step 2:

| Metric | reference prompt (gateway, 1024 tokens) | background prompt (local, 4096 tokens) |
|---|---|---|
| Facts per candidate (median) | 14 | 11 |
| Answer not parsed | 1.4% | 0.0% |
| Mean accuracy | 71.2% | 73.3% |
| Majority vote | 72.8% | 76.4% |
| pass@10 | 85.6% | 88.6% |
| pass@50 | 89.4% | 93.0% |
| >=90% of candidates agree on a wrong answer | 12.4% | 9.9% |

Step 3, first candidate whose facts all pass:

| Candidates | Judge | First valid right | Facts rejected | Accuracy if passed / if rejected |
|---|---|---|---|---|
| reference | glm-5.3-flash-local (question and options) | 77.0% | 7.8% | 76.8% / 44.8% |
| reference | gemini-3.8-flash (question and options) | 75.2% | 11.6% | 76.4% / 40.1% |
| reference | deepseek-v4-flash-think-high (question and options) | 72.7% | 8.8% | - |
| background | glm-5.3-flash-local-fact-only | 74.3% | 10.2% | 75.6% / 67.9% |

Single-shot gemma-4-26b on the same questions: about 68.7%.

## Findings

- The background prompt improves generation on every metric, most on
  majority vote (+3.6 pp), mainly by locking onto a shared wrong answer less
  often.
- The question-unaware judge adds nothing over majority vote on these
  candidates (74.3% vs 76.4%): its rejections carry little signal about the
  answer (75.6% vs 67.9%), because the candidates' errors are mostly in
  applying true facts to the case, which it does not see. Judging only where
  votes split (top option below 60-80%) recovers the vote's accuracy at 14-26%
  of the judge calls.
- The judge that sees the question does separate right from wrong candidates,
  partly by solving the question itself.

## Caveats

- The two step 2 runs differ in more than the prompt: max_tokens 1024 vs 4096
  and gateway vs local serving. A reference-prompt run on local gemma at 4096
  tokens would isolate the prompt's share.
- Judges stop at the first valid candidate, about 4 of 50 per question, so the
  pass/reject accuracies are not over a random sample of candidates.
