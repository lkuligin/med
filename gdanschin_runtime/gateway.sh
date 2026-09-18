# Loads LLM-gateway settings for anything that calls it. Sourced, never executed.
#
# Precedence, highest first:
#   1. LLM_GATEWAY_URL / LLM_GATEWAY_TOKEN environment variables
#   2. configs/gateway.conf  (gitignored; copy configs/gateway.conf.example)
#
# Exports the per-provider proxy bases too, since that is what an
# OpenAI-compatible client actually needs as its api_base.

_GATEWAY_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_GATEWAY_CONF="$_GATEWAY_SH_DIR/configs/gateway.conf"

if [[ -f "$_GATEWAY_CONF" ]]; then
    # shellcheck source=configs/gateway.conf
    source "$_GATEWAY_CONF"
fi

LLM_GATEWAY_URL="${LLM_GATEWAY_URL:-}"
LLM_GATEWAY_TOKEN="${LLM_GATEWAY_TOKEN:-}"

if [[ -z "$LLM_GATEWAY_URL" || -z "$LLM_GATEWAY_TOKEN" || "$LLM_GATEWAY_TOKEN" == "paste-token-here" ]]; then
    echo "LLM gateway not configured (LLM_GATEWAY_URL, LLM_GATEWAY_TOKEN)" >&2
    echo "  cp gdanschin_runtime/configs/gateway.conf.example gdanschin_runtime/configs/gateway.conf" >&2
    echo "  then paste the token, or set the matching LLM_GATEWAY_* variables" >&2
    return 1 2>/dev/null || exit 1
fi

LLM_GATEWAY_URL="${LLM_GATEWAY_URL%/}"   # tolerate a trailing slash
GATEWAY_OPENAI_BASE="$LLM_GATEWAY_URL/proxy/openai"
GATEWAY_ANTHROPIC_BASE="$LLM_GATEWAY_URL/proxy/anthropic"
GATEWAY_GOOGLE_BASE="$LLM_GATEWAY_URL/proxy/google"

export LLM_GATEWAY_URL LLM_GATEWAY_TOKEN
export GATEWAY_OPENAI_BASE GATEWAY_ANTHROPIC_BASE GATEWAY_GOOGLE_BASE
unset _GATEWAY_CONF _GATEWAY_SH_DIR
