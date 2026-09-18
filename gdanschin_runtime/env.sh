# Opt-ins for running llm_monkeys our way. Source before invoking anything:
#
#   source gdanschin_runtime/env.sh
#
# Every variable here is off by default upstream, so an unsourced shell
# reproduces the original authors' behaviour exactly.

# Route difficult-question output to a candidate file and refuse writes to the
# tracked difficult_questions.csv. The path is gitignored and excluded from the
# GPU-box sync, so a candidate can neither be committed nor shipped by accident.
# Promote with ./gdanschin_runtime/promote_difficult_questions.sh
export MEDQA_DIFFICULT_CANDIDATE="data/difficult_questions.candidate.csv"

# Redirect model construction to our own client instead of the stock LiteLlm
# adapter. Commented out until adapters/factory.py exists: an unimportable spec
# fails loudly by design, which would break the stock CLIs.
# export MEDQA_MODEL_FACTORY="gdanschin_runtime.adapters.factory:build"
