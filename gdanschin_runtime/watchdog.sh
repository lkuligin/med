#!/usr/bin/env bash
# Restart the overnight plan if it has stopped without finishing.
#
# Meant for cron, every ten minutes:
#   */10 * * * * /home/gdanschin/Projects/med/gdanschin_runtime/watchdog.sh
#
# Restarting costs nothing: every step resumes from what is stored, so a plan
# that starts again from stage 1 skips the questions already answered, the
# candidates already generated and the verdicts already reached, and picks up
# exactly where the run died.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/logs/plan.log"
BEAT="$REPO/logs/watchdog.log"

# A heartbeat on every run, whatever it decides. A watchdog that only writes
# when it acts cannot be told apart from one cron never calls.
mkdir -p "$REPO/logs"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  watchdog: $1" >> "$BEAT"; }

[[ -f "$LOG" ]] || { say "no plan has ever run"; exit 0; }

# Nothing to do once the plan reports it is done - the watchdog must not start
# the whole thing over after a successful finish.
if [[ "$(grep -E "PLAN (starts|finished)" "$LOG" | tail -1)" == *"PLAN finished"* ]]; then
    say "the plan has finished, nothing to watch"
    exit 0
fi

if pgrep -f "[r]un_plan.py" >/dev/null; then
    say "the plan is running ($(grep -c "STAGE" "$LOG") stage lines so far)"
    exit 0
fi

say "the plan is NOT running, restarting it"
echo "$(date '+%Y-%m-%d %H:%M:%S')  watchdog: the plan is not running, restarting it" >> "$LOG"
"$REPO/gdanschin_runtime/run_plan.sh" >> "$BEAT" 2>&1
