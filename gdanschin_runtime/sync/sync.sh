#!/usr/bin/env bash
# Mirror this repository to the GPU box, one way.
#
#   ./gdanschin_runtime/sync/sync.sh              watch and sync until Ctrl-C
#   ./gdanschin_runtime/sync/sync.sh --once       sync once and exit
#   ./gdanschin_runtime/sync/sync.sh --dry-run    show what would be sent, transfer nothing
#   ./gdanschin_runtime/sync/sync.sh --push-secrets  push the token files, once
#   ./gdanschin_runtime/sync/sync.sh --pull notebooks/  bring remote work home
#
# Token files are kept out of the mirror on purpose: rsync -a preserves
# permissions, and a 644 secret on a shared GPU box is readable by everyone
# there. --push-secrets copies them deliberately and chmods 600 on arrival.
# The remote needs it to reach the gateway, so push it once after creating or
# rotating the token; being excluded, it is also safe from --delete and stays.
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

if [[ "${1:-}" == "--pull" ]]; then
    # The mirror only ever pushes, and paths like notebooks/ are excluded from
    # it entirely, so work done on the box has no way home without this.
    SUB="${2:-}"
    [[ -n "$SUB" ]] || { echo "usage: sync.sh --pull <path relative to the repo>" >&2; exit 2; }
    echo "pulling $SUB from $REMOTE_HOST"
    # No --delete: this must never remove local files. Excludes still apply, so
    # a pull cannot drag caches or a remote .venv back with it.
    rsync -az --exclude-from="$EXCLUDES" -e "ssh -i $IDENTITY" \
        "$REMOTE:$REMOTE_DIR/${SUB%/}/" "$SOURCE/${SUB%/}/"
    echo "into ${SOURCE%/}/${SUB%/}/"
    exit 0
fi

if [[ "${1:-}" == "--push-secrets" ]]; then
    ssh -i "$IDENTITY" "$REMOTE" "mkdir -p '$REMOTE_DIR/gdanschin_runtime/configs'"
    pushed=0
    for name in gateway.conf jupyter-password.conf; do
        src="$HERE/../configs/$name"
        [[ -f "$src" ]] || { echo "  skipping $name (not present here)"; continue; }
        if grep -qE 'paste-token-here|paste-hash-here' "$src"; then
            echo "  skipping $name (still holds a placeholder)" >&2
            continue
        fi
        echo "  pushing $name"
        rsync -a -e "ssh -i $IDENTITY" "$src" "$REMOTE:$REMOTE_DIR/gdanschin_runtime/configs/$name"
        # chmod is a separate step: the rsync macOS ships does not accept --chmod.
        ssh -i "$IDENTITY" "$REMOTE" "chmod 600 '$REMOTE_DIR/gdanschin_runtime/configs/$name'"
        pushed=$((pushed + 1))
    done
    [[ $pushed -gt 0 ]] && ssh -i "$IDENTITY" "$REMOTE" "ls -l '$REMOTE_DIR/gdanschin_runtime/configs/'*.conf"
    exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "DRY RUN: $SOURCE/  ->  $REMOTE:$REMOTE_DIR/"
    echo "Nothing is transferred and nothing on the remote is touched."
    exec rsync -azn --delete --itemize-changes --exclude-from="$EXCLUDES" \
        -e "ssh -i $IDENTITY" "$SOURCE/" "$REMOTE:$REMOTE_DIR/"
fi

# Directories the remote owns. They are excluded from the mirror, which means
# rsync will never create them either, so the box would otherwise start out
# without the very places work is supposed to be written. Keep this list in step
# with the REMOTE-OWNED section of rsync-exclude.txt.
REMOTE_OWNED_DIRS=(data results cache logs scratch notebooks)

# rsync creates the leaf directory, but not missing parents. Done once at
# startup rather than per push: the watch loop syncs on every file save, and an
# extra ssh round-trip each time would be paid thousands of times a day.
echo "Ensuring $REMOTE_DIR and its working directories exist on $REMOTE_HOST..."
mkdirs="mkdir -p '$REMOTE_DIR'"
for d in "${REMOTE_OWNED_DIRS[@]}"; do
    mkdirs="$mkdirs '$REMOTE_DIR/$d'"
done
ssh -i "$IDENTITY" "$REMOTE" "$mkdirs"

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
