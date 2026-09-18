# LLM Monkeys - Medical Inference Scaling

Implementation of inference scaling approaches in the medical domain inspired by the paper [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787).

## Step 1: One-Shot Inference Workflow with Gemma 4 on Vertex MaaS

This workflow loads the `test` split of the MedQA dataset (`bigbio/med_qa`, config `med_qa_en_source`), constructs one-shot inference prompts, invokes Gemma 4 on Vertex AI MaaS (`vertex_ai/google/gemma-4-26b-a4b-it-maas`) using the Google Agent Development Kit (ADK) `LiteLlm` adapter, and outputs structured evaluation results to a JSON file.

### Usage

Run one-shot inference on the MedQA test set:

```bash
# Run on sample of questions
python3 run_one_shot_inference.py --limit 10 --concurrency 2 --output results_sample.json

# Run full evaluation on test split
python3 run_one_shot_inference.py --split test --output results_medqa_test_gemma4.json
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | `vertex_ai/google/gemma-4-26b-a4b-it-maas` | Vertex AI model name |
| `--dataset` | str | `bigbio/med_qa` | Hugging Face dataset name |
| `--dataset-config` | str | `med_qa_en_source` | Dataset configuration |
| `--split` | str | `test` | Dataset split (`test`, `validation`, `train`) |
| `--limit` | int | `None` | Max number of questions to evaluate |
| `--offset` | int | `0` | Offset to start evaluation from |
| `--n-attempts` | int | `3` | Number of inference attempts per question ($n \ge 3$) |
| `--concurrency` | int | `2` | Max concurrent requests |
| `--max-parse-retries` | int | `3` | Max retries if predicted option is unparsed (None) |
| `--output` | str | `results_one_shot_gemma4.json` | Path to save output JSON |
| `--temperature` | float | `0.0` | Sampling temperature |

### Running Results Separation

Separate question IDs answered correctly across ALL attempts by ALL models:

```bash
# Analyze multiple experiment results and extract all-correct question IDs into CSV
python3 separate_result.py results_gemma4.json results_gemini.json results_gpt-oss-20b.json \
      --output simple_questions.csv \
      --difficult-output difficult_questions.csv

# Optional: set MEDQA_DIFFICULT_CANDIDATE to a path to make that path the
# default output and refuse writes to the tracked difficult_questions.csv.
# Useful when this repository is mirrored to another machine, where a write to
# a synced file would be reverted by the next sync pass. Unset, the command
# above writes difficult_questions.csv directly, as it always has.

# Include question text and ground truth in CSVs
python3 separate_result.py results_gemma4.json results_gemini.json results_gpt-oss-20b.json \
      --include-metadata \
      --output simple_questions_with_meta.csv
```

## Step 2: Candidate-Based Repeated Sampling Workflow (`inference/cli.py`)

This workflow implements Step 2 of the research flow: repeated sampling ($N=1000$ candidates) on complex/difficult questions identified from Step 1.

Each candidate reasoning path is generated using a two-step agentic process:
1. **Medical Facts Generation**: Generates atomic, verifiable clinical statements relevant to answering the question (structured output).
2. **Final Answer Generation**: Deduces the final answer option (A, B, C, D) conditioned on the generated medical facts.

The workflow executes asynchronously with semaphore-managed concurrency (`asyncio.Semaphore`), exponential backoff with jitter on 429 rate limits, and saves progress incrementally. Rich metadata is captured for every candidate (prompt tokens, candidate tokens, total tokens, execution latency, raw responses, and correctness against ground truth).

### Usage

Run candidate inference on difficult questions:

```bash
# Quick smoke test on 1 question with 5 candidates
python3 inference/cli.py \
      --difficult-questions difficult_questions.csv \
      --limit 1 \
      --n-candidates 5 \
      --output test_candidates.json

# Full Step 2 run with Gemma 4 on Vertex AI (N=1000 candidates per question)
python3 inference/cli.py \
      --project "kuligin-sandbox1" \
      --location "global" \
      --difficult-questions difficult_questions.csv \
      --n-candidates 1000 \
      --concurrency 4 \
      --output results_step2_gemma4_candidates.json

# Run with GPT-OSS 20B
python3 inference/cli.py \
      --model gpt-oss-20b \
      --project "kuligin-sandbox1" \
      --location "global" \
      --difficult-questions difficult_questions.csv \
      --n-candidates 1000 \
      --concurrency 4 \
      --output results_step2_gpt20b_candidates.json

# Run with Gemini Flash 3.8
python3 inference/cli.py \
      --model gemini-3.8-flash \
      --project "kuligin-sandbox1" \
      --location "global" \
      --difficult-questions difficult_questions.csv \
      --n-candidates 1000 \
      --concurrency 4 \
      --output results_step2_gemini_candidates.json
```

You can also run the CLI as a Python module:

```bash
python3 -m inference.cli --help
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--difficult-questions` | str | `difficult_questions.csv` | Path to CSV containing difficult question IDs |
| `--n-candidates`, `-n` | int | `1000` | Number of reasoning candidate paths to generate per question |
| `--model` | str | `vertex_ai/google/gemma-4-26b-a4b-it-maas` | Vertex AI model name or alias (`gemma-4-26b`, `gpt-oss-20b`, `gemini-3.8-flash`) |
| `--concurrency` | int | `4` | Maximum concurrent model requests (`asyncio.Semaphore`) |
| `--temperature` | float | `0.8` | Sampling temperature for repeated sampling |
| `--save-every-n-candidates` | int | `25` | Save results incrementally every N candidates per question |
| `--save-every-n-questions` | int | `1` | Save results after every N completed questions |
| `--output` | str | `results_step2_gemma4_candidates.json` | Path to save output JSON |
| `--limit` | int | `None` | Maximum number of difficult questions to evaluate |
| `--offset` | int | `0` | Starting index offset in difficult questions list |
| `--dataset` | str | `bigbio/med_qa` | Hugging Face dataset name |
| `--dataset-config` | str | `med_qa_en_source` | Dataset configuration |
| `--split` | str | `test` | Dataset split (`test`, `validation`, `train`) |
| `--max-tokens` | int | `1024` | Maximum output tokens per generation step |
| `--project` | str | `None` | GCP Project ID (or set `GOOGLE_CLOUD_PROJECT` env var) |
| `--location` | str | `global` | Vertex AI location (or set `VERTEXAI_LOCATION` env var) |
| `--max-retries` | int | `5` | Maximum retries for model invocation errors |
| `--rate-limit-max-retries` | int | `10` | Maximum retries with exponential backoff on 429 rate limits |
| `--log-level` | str | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Step 3: Fact Verification Workflow (`verifier/cli.py`)

This workflow implements Step 3 of the research flow: fact verification of candidate reasoning paths generated in Step 2 using an LLM-as-a-judge (`gemini-3-flash-preview` on Vertex AI via Google ADK).

For each complex question, the verifier evaluates candidate reasoning paths one-by-one:
1. **Fact Verification**: Each atomic medical fact in the candidate is verified for clinical correctness against the question context.
2. **First-Valid Selection (Rejection Sampling)**: The first candidate where **all** facts are verified as correct is selected as the final answer. If a candidate contains any incorrect fact, the workflow advances to the next candidate.

The workflow executes asynchronously with semaphore-managed concurrency (`asyncio.Semaphore`), exponential backoff with jitter on 429 rate limits, and resumes cleanly from previously saved runs.

### Usage

Run fact verification on Step 2 candidates:

```bash
# Quick smoke test on 1 question with candidate limit and early stopping
python3 verifier/cli.py \
      --input results_step2_gemma4_candidates.json \
      --limit 1 \
      --max-candidates-per-question 5 \
      --early-stop-facts \
      --output test_verified.json

# Full Step 3 verification run with Gemini Flash on Vertex AI
python3 verifier/cli.py \
      --project "kuligin-sandbox1" \
      --location "global" \
      --input results_step2_gemma4_candidates.json \
      --concurrency 4 \
      --early-stop-facts \
      --output results_step3_verified.json

# Run verification with custom model
python3 verifier/cli.py \
      --model "gemini-3-flash-preview" \
      --input results_step2_gemma4_candidates.json \
      --output results_step3_verified.json
```

You can also run the CLI as a Python module:

```bash
python3 -m verifier.cli --help
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input`, `-i` | str | `results_step2_gemma4_candidates.json` | Path to Step 2 JSON candidate output file |
| `--output` | str | `results_step3_verified.json` | Path to save output JSON |
| `--model` | str | `gemini-3-flash-preview` | Verifier LLM-as-a-judge model identifier |
| `--concurrency` | int | `4` | Maximum concurrent verification requests (`asyncio.Semaphore`) |
| `--early-stop-facts` | bool | `False` | Stop evaluating remaining facts for a candidate upon the first incorrect fact |
| `--max-candidates-per-question` | int | `None` | Maximum candidates to evaluate per question before stopping (default: evaluate all until first valid) |
| `--save-every-n-questions` | int | `1` | Persist intermediate results after every N questions |
| `--limit` | int | `None` | Maximum number of questions to evaluate |
| `--offset` | int | `0` | Starting index offset in questions list |
| `--temperature` | float | `1.0` | Sampling temperature for verifier |
| `--max-tokens` | int | `512` | Maximum output tokens per verification call |
| `--project` | str | `None` | GCP Project ID (or set `GOOGLE_CLOUD_PROJECT` env var) |
| `--location` | str | `global` | Vertex AI location (or set `VERTEXAI_LOCATION` env var) |
| `--max-retries` | int | `5` | Maximum retries for model invocation errors |
| `--rate-limit-max-retries` | int | `10` | Maximum retries with exponential backoff on 429 rate limits |
### Verifier Analysis & Scaling Curves (`analyze_verifier.py`)

Analyze verifier output files and plot the percentage of questions answered correctly after $k = 1, 2, \dots$ candidate generations. A question is counted as correctly answered when it has passed all verification checks AND the candidate's answer matches ground truth.

```bash
# Analyze results and generate plot
python3 analyze_verifier.py --input results_step3_verified.json --output verifier_accuracy_curve.png

# Compare Verified & Correct against Rejection Sampling (First Valid) and Unverified Baseline
python3 analyze_verifier.py --input results_step3_verified.json --strategy all --output verifier_comparison.png

# Export curve data to CSV and metrics to JSON
python3 analyze_verifier.py --input results_step3_verified.json \
      --export-csv verifier_curve.csv \
      --export-json verifier_metrics.json

# Analyze up to k=50 candidates with logarithmic scale
python3 analyze_verifier.py --input results_step3_verified.json --max-candidates 50 --log-scale-x
```

### CLI Arguments for `analyze_verifier.py`

| Argument | Type | Default | Description |
|---|---|---|---|
| `positional_input` / `--input`, `-i` | str | `results_step3_verified.json` | Path to verifier JSON results file |
| `--output`, `-o` | str | `verifier_accuracy_curve.png` | Path to save accuracy curve plot (PNG/PDF/SVG) |
| `--strategy`, `-s` | str | `both` | Plot mode: `verified` (primary metric), `first_valid` (rejection sampling), `both`, or `all` |
| `--max-candidates`, `-k` | int | `None` | Max candidate count $k$ to analyze up to (default: all available) |
| `--min-candidates` | int | `1` | Starting candidate count $k$ |
| `--step` | int | `1` | Step size between candidate checkpoints |
| `--log-scale-x` | flag | `False` | Use logarithmic scale for x-axis |
| `--no-plot` | flag | `False` | Skip plot generation and print ASCII table & summary |
| `--export-csv` | str | `None` | Optional path to export curve table to CSV |
| `--export-json` | str | `None` | Optional path to export metrics and cost to JSON |
| `--model` | str | `gemini-3-flash-preview` | Model name for verifier token cost calculation |

### Running Tests

```bash
pytest
```


