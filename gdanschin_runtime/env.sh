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

# Serve every agent through the LLM gateway instead of Vertex AI. Without this
# the stock CLIs reach for Vertex credentials we do not have.
export MEDQA_MODEL_FACTORY="gdanschin_runtime.adapters.factory:build"

# ADK suggests its native Gemini integration whenever a gemini model goes
# through LiteLLM. Here it has to: the gateway speaks the LiteLLM dialect.
export ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS=true
