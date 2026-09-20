#!/usr/bin/env bash
# Run step 2, with step 3 following a question or two behind.
#
#   ./gdanschin_runtime/pipeline.sh --base gemma-4-26b --limit 483 --n-candidates 20
#   ./gdanschin_runtime/pipeline.sh status [--base NAME]
#   ./gdanschin_runtime/pipeline.sh stop
#
# --base and --judge name entries in gdanschin_runtime/models.py, which decide
# both the model called and the directory results land in. Other arguments pass
# through to inference.cli.
#
# Step 3 reads the candidates straight out of the store, so it needs no copy of
# step 2's output and can run while step 2 is still generating: a record is
# written once and never revised, so what it reads is always whole.
#
# Detached with setsid rather than tmux. tmux hands a new session the
# environment of the tmux SERVER - started long ago by whoever opened the first
# session - not of the shell that launched it, so MEDQA_MODEL_FACTORY never
# arrived and the run silently reached for Vertex AI. setsid inherits the
# environment it was called with and survives the ssh connection just as well.
#
# The steps use different gateways on purpose: generation goes to the internal
# one, which showed no rate limit and saturates around 8 concurrent, judging to
# the external one, where Gemini lives behind a per-user limit on requests in
# flight, so it stays low and simply lags behind.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="$REPO/.venv/bin/python"
MONKEYS="$REPO/llm_monkeys"
LOGS="$REPO/logs"

BASE="${MEDQA_BASE_MODEL:-gemma-4-26b}"
JUDGE="${MEDQA_JUDGE_MODEL:-gemini-3.8-flash}"
COMMAND=""
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)  BASE="$2"; shift 2 ;;
        --judge) JUDGE="$2"; shift 2 ;;
        status|stop) COMMAND="$1"; shift ;;
        *) PASS+=("$1"); shift ;;
    esac
done
export MEDQA_BASE_MODEL="$BASE" MEDQA_JUDGE_MODEL="$JUDGE"

GEN_CONCURRENCY="${GEN_CONCURRENCY:-8}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-2}"
CYCLE_SECONDS="${CYCLE_SECONDS:-180}"

if [[ "$COMMAND" == "status" ]]; then
    # Step 1 is not started from here, but its results share the run name, so
    # a status for this base is only half a status without it.
    one_shot=$(pgrep -f "[r]un_step1.py|[-]m cli" >/dev/null && echo running || echo idle)
    gen=$(pgrep -f "[i]nference.cli" >/dev/null && echo running || echo stopped)
    judge=$(pgrep -f "[v]erifier.cli" >/dev/null && echo running || echo idle)
    pos=$(grep -o "Processing Question [0-9]*/[0-9]*" "$LOGS/step2.log" 2>/dev/null | tail -1)
    echo "  base       $BASE   judge $JUDGE"
    echo "  one shot   $one_shot"
    echo "  generation $gen   ${pos:-}"
    echo "  judging    $judge"
    "$PY" -c "
import sys
sys.path.insert(0, '$REPO')
from gdanschin_runtime import _bootstrap
from results_store import CandidateResults, OneShotResults, VerificationResults
one_shot = OneShotResults('$MONKEYS/results', '$BASE')
candidates = CandidateResults('$MONKEYS/results', '$BASE')
verdicts = VerificationResults('$MONKEYS/results', '$BASE', '$JUDGE')
qs = candidates.questions()
answered = len(list(one_shot.directory.glob('question_*.json'))) if one_shot.directory.is_dir() else 0
print(f'  answered   {answered} questions one shot')
print(f'  generated  {len(qs)} questions, {sum(candidates.candidate_count(q) for q in qs)} candidates')
print(f'  judged     {sum(1 for q in qs if verdicts.judge_dir(q).is_dir())} questions')"
    [[ -f "$LOGS/step2.log" ]] && echo "  gen errors $(grep -c Exhausted "$LOGS/step2.log")"
    exit 0
fi

if [[ "$COMMAND" == "stop" ]]; then
    pkill -f "[i]nference.cli" 2>/dev/null && echo "  generation stopped" || echo "  generation was not running"
    pkill -f "[v]erifier.cli" 2>/dev/null && echo "  judging stopped" || echo "  judging was not running"
    pkill -f "[p]ipeline.sh" 2>/dev/null || true
    exit 0
fi

# shellcheck source=env.sh
source "$HERE/env.sh"
export MEDQA_BASE_MODEL="$BASE" MEDQA_JUDGE_MODEL="$JUDGE"
mkdir -p "$LOGS"

echo "base       $BASE   judge $JUDGE"
echo "generation -> results/facts-pipeline/$BASE   internal gateway, concurrency $GEN_CONCURRENCY"
cd "$MONKEYS"
setsid nohup "$PY" -m inference.cli --concurrency "$GEN_CONCURRENCY" \
    --run-name "$BASE" "${PASS[@]}" > "$LOGS/step2.log" 2>&1 < /dev/null &
echo "  pid $!"

# --judge-name only names the directory verdicts are stored in; the model and
# the temperature have to be handed over too, or the verifier quietly keeps its
# own defaults whatever the name promises.
read -r JUDGE_MODEL JUDGE_TEMPERATURE JUDGE_MAX_TOKENS < <("$PY" -c "
import sys
sys.path.insert(0, '$REPO')
from gdanschin_runtime.models import JUDGE_MODELS
j = JUDGE_MODELS['$JUDGE']
print(j.gateway_model, j.temperature, j.max_tokens)")
# Trying a different temperature takes an environment variable rather than a
# new entry in models.py - but then name the judge too, or its verdicts land
# among the ones it is meant to be compared with.
JUDGE_TEMPERATURE="${MEDQA_JUDGE_TEMPERATURE:-$JUDGE_TEMPERATURE}"

echo "judging    -> $JUDGE ($JUDGE_MODEL at temperature $JUDGE_TEMPERATURE)   external gateway, concurrency $JUDGE_CONCURRENCY"
setsid nohup bash -c '
    while true; do
        generating=$(pgrep -f "[i]nference.cli" >/dev/null && echo yes || echo no)
        "'"$PY"'" -m verifier.cli --run-name "'"$BASE"'" --judge-name "'"$JUDGE"'" \
            --model "'"$JUDGE_MODEL"'" --temperature '"$JUDGE_TEMPERATURE"' \
            --max-tokens '"$JUDGE_MAX_TOKENS"' \
            --concurrency '"$JUDGE_CONCURRENCY"' >> "'"$LOGS"'/step3.log" 2>&1 || true
        [[ "$generating" == "no" ]] && break
        sleep '"$CYCLE_SECONDS"'
    done
' > "$LOGS/watch.log" 2>&1 < /dev/null &
echo "  pid $!"
echo
echo "watch it with:  ./gdanschin_runtime/pipeline.sh status --base $BASE"
