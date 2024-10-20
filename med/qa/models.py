import json

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_vertexai import (
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory,
    VertexAIModelGarden,
    get_vertex_maas_model,
)
from langchain_google_vertexai.gemma import GemmaChatVertexAIModelGarden
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

_GEMINI_MODELS = [
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-002",
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-1.5-flash-001",
    "gemini-pro-experimental",
]


rate_limiter_llama = InMemoryRateLimiter(requests_per_second=1.0)
rate_limiter_llama2 = InMemoryRateLimiter(requests_per_second=2.0)
rate_limiter_mistral = InMemoryRateLimiter(requests_per_second=2.0)
rate_limiter_exp = InMemoryRateLimiter(requests_per_second=0.5)


def get_model(model_name: str, temperature: float = 0.0, **kwargs):
    """Prepares a chat model for experiments."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
    with open("config.json", "r") as read_f:
        config = json.load(read_f)
    if model_name in _GEMINI_MODELS:
        return ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            safety_settings=safety_settings,
            **kwargs,
        )
    if model_name in ["gemma_9b_it", "gemma_27b_it"]:
        llm = VertexAIModelGarden(
            endpoint_id=config["models"][model_name]["endpoint_id"],
            project="kuligin-sandbox1",
            location=config["models"][model_name]["location"],
            prompt_arg="inputs",
            allowed_model_args=["temperature", "max_tokens"],
        ).bind(temperature=temperature)
        if "max_output_tokens" in kwargs:
            return llm.bind(max_tokens=kwargs["max_output_tokens"])
        return llm
    if model_name in ["gemma_2b", "gemma_2b_it"]:
        llm = GemmaChatVertexAIModelGarden(
            endpoint_id=config["models"][model_name]["endpoint_id"],
            project="kuligin-sandbox1",
            location=config["models"][model_name]["location"],
            temperature=temperature,
        )
        if "max_output_tokens" in kwargs:
            return llm.bind(max_tokens=kwargs["max_output_tokens"])
        return llm
    if model_name in ["llama_2b"]:
        llm = VertexAIModelGarden(
            endpoint_id=config["models"][model_name]["endpoint_id"],
            project="kuligin-sandbox1",
            location=config["models"][model_name]["location"],
            allowed_model_args=["temperature", "max_tokens"],
        ).bind(temperature=temperature)
        if "max_output_tokens" in kwargs:
            return llm.bind(max_tokens=kwargs["max_output_tokens"])
        return llm
    if model_name in ["medllama3"]:
        llm = VertexAIModelGarden(
            endpoint_id=config["models"][model_name]["endpoint_id"],
            project="kuligin-sandbox1",
            location=config["models"][model_name]["location"],
            allowed_model_args=["temperature", "max_tokens"],
            prompt_arg="inputs",
        ).bind(temperature=temperature)
        if "max_output_tokens" in kwargs:
            return llm.bind(max_tokens=kwargs["max_output_tokens"])
        return llm
    if model_name == "llama_3_405b":
        return get_vertex_maas_model(
            model_name="meta/llama3-405b-instruct-maas",
            temperature=temperature,
            rate_limiter=rate_limiter_llama,
            append_tools_to_system_message=True,
        )
    if model_name == "llama_3.2_90b":
        return get_vertex_maas_model(
            model_name="meta/llama-3.2-90b-vision-instruct-maas",
            temperature=temperature,
            rate_limiter=rate_limiter_llama2,
            append_tools_to_system_message=True,
        )
    if model_name == "mistral_large":
        return get_vertex_maas_model(
            model_name="mistral-large@2407",
            temperature=temperature,
            rate_limiter=rate_limiter_mistral,
        )
    if model_name == "mistral_nemo":
        return get_vertex_maas_model(
            model_name="mistral-nemo@2407",
            temperature=temperature,
            rate_limiter=InMemoryRateLimiter(requests_per_second=2.0),
        )
    if model_name == "anthropic_claude":
        return ChatAnthropicVertex(
            model_name="claude-3-5-sonnet@20240620",
            temperature=temperature,
            rate_limiter=InMemoryRateLimiter(requests_per_second=2.0),
            location="us-east5",
        )
