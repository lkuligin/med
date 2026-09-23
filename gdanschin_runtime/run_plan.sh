#!/usr/bin/env bash
# Start the overnight plan, detached, with our environment in place.
#
#   ./gdanschin_runtime/run_plan.sh            # from the beginning
#   ./gdanschin_runtime/run_plan.sh --from 3   # resume at stage 3
#   ./gdanschin_runtime/run_plan.sh --list     # what will run
#   ./gdanschin_runtime/run_plan.sh stop
#
# setsid rather than tmux: a tmux session inherits the tmux server's
# environment, not the shell's, and that once sent a whole run to Vertex AI.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

if [[ "${1:-}" == "stop" ]]; then
    # The driver first, so it cannot start the next step while the current one
    # is being killed - then every step it can have started. run_step1.py
    # belongs in this list: leaving it out once left it running alone, still
    # calling the gateway, after everything else had stopped.
    pkill -f "[r]un_plan.py" 2>/dev/null && echo "  plan stopped" || echo "  plan was not running"
    sleep 1
    for pattern in "[r]un_step1.py" "[i]nference.cli" "[v]erifier.cli"; do
        pkill -f "$pattern" 2>/dev/null && echo "  killed ${pattern//[\[\]]/}" || true
    done
    sleep 1
    if pgrep -f "[r]un_plan.py|[r]un_step1.py|[i]nference.cli|[v]erifier.cli" >/dev/null; then
        echo "  still running:" >&2
        pgrep -af "[r]un_plan.py|[r]un_step1.py|[i]nference.cli|[v]erifier.cli" >&2
        exit 1
    fi
    echo "  nothing is running"
    exit 0
fi

# shellcheck source=env.sh
source "$HERE/env.sh"

if [[ "${1:-}" == "--list" ]]; then
    exec "$REPO/.venv/bin/python" "$HERE/run_plan.py" --list
fi

if pgrep -f "[r]un_plan.py" >/dev/null; then
    echo "a plan is already running; stop it first" >&2
    exit 1
fi

mkdir -p "$REPO/logs"
cd "$REPO"
setsid nohup "$REPO/.venv/bin/python" "$HERE/run_plan.py" "$@" \
    >> "$REPO/logs/plan.out" 2>&1 < /dev/null &
echo "  started, pid $!"
echo "  follow it with:  tail -f logs/plan.log"
