#!/usr/bin/env bash
# Run the test suite here and on the GPU box, and report both.
#
#   ./gdanschin_runtime/run_tests.sh              both machines
#   ./gdanschin_runtime/run_tests.sh --local      here only
#   ./gdanschin_runtime/run_tests.sh --remote     GPU box only
#   ./gdanschin_runtime/run_tests.sh --no-sync    do not push changes first
#   ./gdanschin_runtime/run_tests.sh -k parser -x anything else goes to pytest
#
# The remote run is pushed to first, otherwise it tests whatever happened to be
# there - which looks like a passing suite while proving nothing about the code
# you just changed.
#
# Exits non-zero if either side fails, so it can gate a commit.
set -uo pipefail   # no -e: a failing suite must still reach the summary

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# shellcheck source=remote.sh
# `|| exit` matters: remote.sh returns non-zero when unconfigured, and this
# script deliberately runs without set -e so a failing suite still reaches the
# summary. Without it we would proceed with an empty host and exit 0.
source "$HERE/remote.sh" || exit 1

# Note the guarded expansions below: macOS still ships bash 3.2, where
# "${arr[@]}" on an EMPTY array counts as an unbound variable under set -u.
RUN_LOCAL=1; RUN_REMOTE=1; DO_SYNC=1; PYTEST_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --local)   RUN_REMOTE=0 ;;
        --remote)  RUN_LOCAL=0 ;;
        --no-sync) DO_SYNC=0 ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         PYTEST_ARGS+=("$arg") ;;
    esac
done

local_status="skipped"; remote_status="skipped"; failed=0

if [[ $RUN_LOCAL -eq 1 ]]; then
    echo "=== local: $(uname -s) $(uname -m) ==="
    if [[ -x "$REPO/.venv/bin/python" ]]; then
        (cd "$REPO/llm_monkeys" && "$REPO/.venv/bin/python" -m pytest ${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"})
        monkeys=$?
        # The runtime has tests of its own - what --base and --judge-name
        # actually select - and they live outside llm_monkeys' testpaths.
        (cd "$REPO" && "$REPO/.venv/bin/python" -m pytest gdanschin_runtime/tests \
            ${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"})
        if [[ $monkeys -eq 0 && $? -eq 0 ]]; then local_status="passed"; else local_status="FAILED"; failed=1; fi
    else
        local_status="no .venv (run setup_env.sh --cpu)"; failed=1
        echo "  $local_status" >&2
    fi
    echo
fi

if [[ $RUN_REMOTE -eq 1 ]]; then
    if [[ $DO_SYNC -eq 1 ]]; then
        echo "=== pushing to $REMOTE_HOST ==="
        if ! "$HERE/sync/sync.sh" --once >/dev/null; then
            echo "  sync failed; the remote run would test stale code" >&2
            remote_status="sync FAILED"; failed=1
            RUN_REMOTE=0
        fi
    fi
fi

if [[ $RUN_REMOTE -eq 1 ]]; then
    echo "=== remote: $REMOTE_HOST ==="
    ssh -i "$REMOTE_IDENTITY" -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" \
        "cd '$REMOTE_DIR/llm_monkeys' && '$REMOTE_DIR/.venv/bin/python' -m pytest ${PYTEST_ARGS[*]+${PYTEST_ARGS[*]}} && \
         cd '$REMOTE_DIR' && '$REMOTE_DIR/.venv/bin/python' -m pytest gdanschin_runtime/tests ${PYTEST_ARGS[*]+${PYTEST_ARGS[*]}}"
    if [[ $? -eq 0 ]]; then remote_status="passed"; else remote_status="FAILED"; failed=1; fi
    echo
fi

echo "================================"
printf "  local   %s\n" "$local_status"
printf "  remote  %s\n" "$remote_status"
echo "================================"
exit $failed
