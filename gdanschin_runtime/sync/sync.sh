#!/usr/bin/env bash
# Mirror this repository to the GPU box, one way.
#
#   ./gdanschin_runtime/sync/sync.sh              watch and sync until Ctrl-C
#   ./gdanschin_runtime/sync/sync.sh --once       sync once and exit
#   ./gdanschin_runtime/sync/sync.sh --dry-run    show what would be sent, transfer nothing
#
# Watching uses fswatch rather than lsyncd: lsyncd's macOS backend opens
# /dev/fsevents directly, which needs root, while fswatch goes through the
# public FSEvents API. sync/lsyncd-med.lua is kept for a Linux host, or for
# running lsyncd under sudo.
#
# This is a true mirror: --delete removes remote files that no longer exist
# locally, so a module deleted here cannot linger there and get imported.
#
# Remote-generated data is protected by rsync-exclude.txt alone. Excluded paths
# are neither transferred nor deleted (--delete-excluded would remove them; we
# never pass it). Write datasets, caches and outputs only into the directories
# listed in that file's REMOTE-OWNED section - data/, results/, cache/, logs/ -
# because anything created outside them WILL be deleted on the next sync.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../remote.sh
source "$HERE/../remote.sh" || exit 1
IDENTITY="$REMOTE_IDENTITY"

SOURCE="${MED_SYNC_SOURCE:-$(cd "$HERE/../.." && pwd)}"
EXCLUDES="$HERE/rsync-exclude.txt"

[[ -f "$IDENTITY" ]] || { echo "identity file not found: $IDENTITY" >&2; exit 1; }
[[ -f "$EXCLUDES" ]] || { echo "exclude list not found: $EXCLUDES" >&2; exit 1; }

# ssh refuses world/group-readable private keys outright.
perms=$(stat -f '%Lp' "$IDENTITY" 2>/dev/null || stat -c '%a' "$IDENTITY")
if [[ "$perms" != "600" && "$perms" != "400" ]]; then
    echo "identity file has permissions $perms; ssh requires 600 or 400" >&2
    echo "  chmod 600 $IDENTITY" >&2
    exit 1
fi

push() {
    rsync -az --delete --exclude-from="$EXCLUDES" -e "ssh -i $IDENTITY" \
        "$SOURCE/" "$REMOTE:$REMOTE_DIR/"
}

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "DRY RUN: $SOURCE/  ->  $REMOTE:$REMOTE_DIR/"
    echo "Nothing is transferred and nothing on the remote is touched."
    exec rsync -azn --delete --itemize-changes --exclude-from="$EXCLUDES" \
        -e "ssh -i $IDENTITY" "$SOURCE/" "$REMOTE:$REMOTE_DIR/"
fi

# rsync creates the leaf directory, but not missing parents.
echo "Ensuring $REMOTE_DIR exists on $REMOTE_HOST..."
ssh -i "$IDENTITY" "$REMOTE" "mkdir -p '$REMOTE_DIR'"

echo "Initial sync..."
push
echo "Initial sync done."

if [[ "${1:-}" == "--once" ]]; then
    exit 0
fi

command -v fswatch >/dev/null || {
    echo "fswatch not installed. On macOS: brew install fswatch" >&2
    exit 1
}

echo "Watching $SOURCE -> $REMOTE_HOST:$REMOTE_DIR  (Ctrl-C to stop)"
# Excluded here too, so that noisy paths do not wake the loop at all. rsync
# still applies the full exclude list to whatever does get transferred.
fswatch -o -r -l 1 \
    --exclude '\.git' \
    --exclude '__pycache__' \
    --exclude '\.pyc$' \
    "$SOURCE" | while read -r _; do
        if push; then
            echo "synced $(date '+%H:%M:%S')"
        else
            echo "sync failed at $(date '+%H:%M:%S'), still watching" >&2
        fi
    done
