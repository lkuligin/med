# Loads the GPU box connection settings for every script that talks to it.
# Sourced, never executed.
#
# Precedence, highest first:
#   1. MED_REMOTE_USER / MED_REMOTE_HOST / MED_REMOTE_DIR / MED_REMOTE_IDENTITY
#   2. configs/remote.conf  (gitignored; copy configs/remote.conf.example)
#
# Nothing is hardcoded here: the host and account belong to whoever is running
# this, not to the repository.

_REMOTE_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REMOTE_CONF="$_REMOTE_SH_DIR/configs/remote.conf"

if [[ -f "$_REMOTE_CONF" ]]; then
    # shellcheck source=configs/remote.conf
    source "$_REMOTE_CONF"
fi

REMOTE_USER="${MED_REMOTE_USER:-${REMOTE_USER:-}}"
REMOTE_HOST="${MED_REMOTE_HOST:-${REMOTE_HOST:-}}"
REMOTE_DIR="${MED_REMOTE_DIR:-${REMOTE_DIR:-}}"
# MED_SYNC_IDENTITY is the older name, still honoured so existing shells keep working.
REMOTE_IDENTITY="${MED_REMOTE_IDENTITY:-${MED_SYNC_IDENTITY:-${REMOTE_IDENTITY:-}}}"

_missing=()
[[ -n "$REMOTE_USER" ]]     || _missing+=("REMOTE_USER")
[[ -n "$REMOTE_HOST" ]]     || _missing+=("REMOTE_HOST")
[[ -n "$REMOTE_DIR" ]]      || _missing+=("REMOTE_DIR")
[[ -n "$REMOTE_IDENTITY" ]] || _missing+=("REMOTE_IDENTITY")

if [[ ${#_missing[@]} -gt 0 ]]; then
    echo "remote settings not configured: ${_missing[*]}" >&2
    echo "  cp gdanschin_runtime/configs/remote.conf.example gdanschin_runtime/configs/remote.conf" >&2
    echo "  then edit it, or set the matching MED_REMOTE_* variables" >&2
    return 1 2>/dev/null || exit 1
fi
unset _missing _REMOTE_CONF _REMOTE_SH_DIR

REMOTE="$REMOTE_USER@$REMOTE_HOST"
