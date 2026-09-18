#!/usr/bin/env bash
# Preflight and start the lsyncd mirror to the GPU box.
#
#   ./gdanschin_runtime/sync/sync.sh              start watching and syncing
#   ./gdanschin_runtime/sync/sync.sh --dry-run    show what a sync would send, transfer nothing
#
# The exclude list lives in rsync-exclude.txt and is shared by both modes.
set -euo pipefail

REMOTE_USER="gdanschin"
REMOTE_HOST="gpu.example.com"
REMOTE_DIR="/home/gdanschin/Projects/med"
IDENTITY="${MED_SYNC_IDENTITY:-$HOME/.ssh/g.danschin}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "DRY RUN: $SOURCE/  ->  $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
    echo "Nothing is transferred and nothing on the remote is touched."
    exec rsync -azn --itemize-changes \
        --exclude-from="$EXCLUDES" \
        -e "ssh -i $IDENTITY" \
        "$SOURCE/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
fi

command -v lsyncd >/dev/null || {
    echo "lsyncd not installed. On macOS: brew install lsyncd" >&2
    exit 1
}

# rsync creates the leaf directory, but not missing parents.
echo "Ensuring $REMOTE_DIR exists on $REMOTE_HOST..."
ssh -i "$IDENTITY" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"

echo "Watching $SOURCE -> $REMOTE_HOST:$REMOTE_DIR  (Ctrl-C to stop)"
export MED_SYNC_CONF_DIR="$HERE" MED_SYNC_SOURCE="$SOURCE" MED_SYNC_IDENTITY="$IDENTITY"
exec lsyncd "$HERE/lsyncd-med.lua"
