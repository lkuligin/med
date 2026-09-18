#!/usr/bin/env bash
# Copy a reviewed candidate over the tracked difficult-questions list.
#
#   ./gdanschin_runtime/promote_difficult_questions.sh [candidate] [target]
#
# Defaults:
#   candidate  llm_monkeys/data/difficult_questions.candidate.csv
#   target     llm_monkeys/difficult_questions.csv
#
# Staging and committing the result is left to you.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="${1:-$REPO/llm_monkeys/data/difficult_questions.candidate.csv}"
TARGET="${2:-$REPO/llm_monkeys/difficult_questions.csv}"

[[ -f "$CANDIDATE" ]] || { echo "candidate not found: $CANDIDATE" >&2; exit 1; }
[[ -s "$CANDIDATE" ]] || { echo "candidate is empty: $CANDIDATE" >&2; exit 1; }

# Count data rows, skipping the header, so the report is about questions.
count_rows() {
    [[ -f "$1" ]] && awk 'NR > 1 && NF' "$1" | wc -l | tr -d ' ' || echo 0
}

before=$(count_rows "$TARGET")
after=$(count_rows "$CANDIDATE")

cp "$CANDIDATE" "$TARGET"

echo "promoted: $CANDIDATE"
echo "      -> $TARGET"
echo "  questions: $before -> $after"
