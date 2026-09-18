#!/usr/bin/env bash
# Take a readable copy of a results file while a run is still writing it.
#
#   ./gdanschin_runtime/snapshot_results.sh [source] [destination]
#
# Defaults to results/step2_gemma_20q_k20.json -> results/snapshot.json, both
# relative to llm_monkeys/. Run this on whichever machine holds the file.
#
# The copy is validated before it replaces the previous snapshot, and retried
# if it does not parse: the pipeline saves after every question, so a plain cp
# can catch a half-written file, and a truncated snapshot looks like a corrupt
# run rather than bad timing.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$REPO/llm_monkeys/results/step2_gemma_20q_k20.json}"
DST="${2:-$REPO/llm_monkeys/results/snapshot.json}"
PY="$REPO/.venv/bin/python"

[[ -f "$SRC" ]] || { echo "no such file: $SRC" >&2; exit 1; }

for attempt in 1 2 3 4 5; do
    cp "$SRC" "$DST.tmp"
    if "$PY" - "$DST.tmp" <<'CHECK_EOF'
import json, sys
data = json.load(open(sys.argv[1]))
results = data["results"] if isinstance(data, dict) and "results" in data else data
candidates = sum(len(q["candidates"]) for q in results)
print(f"  {len(results)} questions, {candidates} candidates")
CHECK_EOF
    then
        mv "$DST.tmp" "$DST"
        echo "  -> ${DST#"$REPO/"}"
        exit 0
    fi
    echo "  attempt $attempt caught a partial write, retrying" >&2
    sleep 3
done

rm -f "$DST.tmp"
echo "could not get a clean copy after 5 attempts" >&2
exit 1
