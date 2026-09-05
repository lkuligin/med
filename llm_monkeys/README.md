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

### Running Tests

```bash
pytest
```

