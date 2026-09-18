#!/usr/bin/env bash
# Run step 2, with step 3 following a question or two behind.
#
#   ./gdanschin_runtime/pipeline.sh --limit 483 --n-candidates 20
#   ./gdanschin_runtime/pipeline.sh status
#   ./gdanschin_runtime/pipeline.sh stop
#
# Other arguments pass through to inference.cli.
#
# Detached with setsid rather than tmux. tmux hands a new session the
# environment of the tmux SERVER - started long ago by whoever opened the first
# session - not of the shell that launched it, so MEDQA_MODEL_FACTORY never
# arrived and the run silently reached for Vertex AI. setsid inherits the
# environment it was called with and survives the ssh connection just as well.
#
# The steps use different gateways on purpose: generation goes to the internal
# one, which showed no rate limit and saturates around 8 concurrent, judging to
# the external one, where Gemini lives behind a per-user RPS limit that bites at
# any concurrency, so it stays low and simply lags behind.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="$REPO/.venv/bin/python"
MONKEYS="$REPO/llm_monkeys"
LOGS="$REPO/logs"

# Results live under results/<model>/ rather than in one shared file. The
# generator merges with whatever output it finds and tops each question up to
# the target count, which is exactly right for resuming an interrupted run and
# exactly wrong across models: a second run would quietly blend another model's
# candidates into the first one's dataset, with nothing to show for it.
RUN_NAME="${RUN_NAME:-}"
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="gemma-4-26b"
    prev=""
    for arg in "$@"; do
        [[ "$prev" == "--model" ]] && RUN_NAME="$arg"
        prev="$arg"
    done
    RUN_NAME="${RUN_NAME//\//-}"
fi
RUN_DIR="$MONKEYS/results/$RUN_NAME"

STEP2_OUT="${STEP2_OUT:-$RUN_DIR/step2.json}"
SNAPSHOT="${SNAPSHOT:-$RUN_DIR/step2_snapshot.json}"
STEP3_OUT="${STEP3_OUT:-$RUN_DIR/step3.json}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-8}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-2}"
CYCLE_SECONDS="${CYCLE_SECONDS:-180}"

count() {
    [[ -f "$1" ]] || { echo 0; return; }
    "$PY" -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    r = d['results'] if isinstance(d, dict) and 'results' in d else d
    print(len(r))
except Exception:
    print(0)" "$1" 2>/dev/null || echo 0
}

case "${1:-}" in
status)
    gen=$(pgrep -f "[i]nference.cli" >/dev/null && echo running || echo stopped)
    judge=$(pgrep -f "[v]erifier.cli" >/dev/null && echo running || echo idle)
    pos=$(grep -o "Processing Question [0-9]*/[0-9]*" "$LOGS/step2.log" 2>/dev/null | tail -1)
    echo "  generation $gen   ${pos:-}"
    echo "  judging    $judge"
    echo "  generated  $(count "$STEP2_OUT") questions"
    echo "  judged     $(count "$STEP3_OUT") questions"
    [[ -f "$LOGS/step2.log" ]] && echo "  gen errors $(grep -c Exhausted "$LOGS/step2.log")"
    exit 0
    ;;
stop)
    pkill -f "[i]nference.cli" 2>/dev/null && echo "  generation stopped" || echo "  generation was not running"
    pkill -f "[v]erifier.cli" 2>/dev/null && echo "  judging stopped" || echo "  judging was not running"
    pkill -f "[p]ipeline.sh watch" 2>/dev/null || true
    exit 0
    ;;
esac

# shellcheck source=env.sh
source "$HERE/env.sh"
mkdir -p "$RUN_DIR" "$LOGS"

echo "run        $RUN_NAME"
echo "generation -> ${STEP2_OUT#"$REPO/"}   internal gateway, concurrency $GEN_CONCURRENCY"
cd "$MONKEYS"
setsid nohup "$PY" -m inference.cli --concurrency "$GEN_CONCURRENCY" \
    --output "$STEP2_OUT" "$@" > "$LOGS/step2.log" 2>&1 < /dev/null &
echo "  pid $!"

echo "judging    -> ${STEP3_OUT#"$REPO/"}   external gateway, concurrency $JUDGE_CONCURRENCY"
setsid nohup bash -c '
    while true; do
        generating=$(pgrep -f "[i]nference.cli" >/dev/null && echo yes || echo no)
        if "'"$HERE"'/snapshot_results.sh" "'"$STEP2_OUT"'" "'"$SNAPSHOT"'" >/dev/null 2>&1; then
            "'"$PY"'" -m verifier.cli --input "'"$SNAPSHOT"'" --output "'"$STEP3_OUT"'" \
                --concurrency '"$JUDGE_CONCURRENCY"' >> "'"$LOGS"'/step3.log" 2>&1 || true
        fi
        [[ "$generating" == "no" ]] && break
        sleep '"$CYCLE_SECONDS"'
    done
' > "$LOGS/watch.log" 2>&1 < /dev/null &
echo "  pid $!"
echo
echo "watch it with:  ./gdanschin_runtime/pipeline.sh status"
