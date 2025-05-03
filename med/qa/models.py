import json
from typing import Any, List, Optional

from google.cloud import aiplatform
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
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
from langchain_openai import ChatOpenAI
from pydantic import Field, model_validator

_GEMINI_MODELS = [
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-002",
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-1.5-flash-001",
    "gemini-pro-experimental",
    "gemini-2.0-flash-001",
]

_GEMINI_MODELS_EXP = ["gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-03-25"]

rate_limiter_llama = InMemoryRateLimiter(requests_per_second=1.0)
rate_limiter_llama2 = InMemoryRateLimiter(requests_per_second=0.5)
rate_limiter_mistral = InMemoryRateLimiter(requests_per_second=2.0)
rate_limiter_exp = InMemoryRateLimiter(requests_per_second=0.5)
rate_limiter_gpt = InMemoryRateLimiter(requests_per_second=0.5)
rate_limiter_gemini_exp = InMemoryRateLimiter(requests_per_second=2.0)


class _DeepSeekLLM(BaseLLM):
    location: str
    project: str
    endpoint_id: str
    client: Any = Field(default=None, exclude=True)  #: :meta private:

    @model_validator(mode="after")
    def validate_environment(self):
        """Validate that the python package exists in environment."""

        self.client = aiplatform.Endpoint(
            self.endpoint_id, project=self.project, location=self.location
        )
        return self

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        instances = [{"text": t} for t in prompts]
        parameters = {
            "sampling_params": {
                "max_new_tokens": 128,
                "temperature": 0.6,
                "top_p": 0.95,
            }
        }
        if "temperature" in kwargs:
            parameters["temperature"] = kwargs["temperature"]
        if "max_output_tokens" in kwargs:
            parameters["max_new_tokens"] = kwargs["max_output_tokens"]

        response = self.client.predict(
            instances=instances, parameters=parameters, use_dedicated_endpoint=True
        )
        generations = []
        for prediction in response.predictions:
            info = prediction["meta_info"]
            usage_metadata = {
                "input_tokens": info["prompt_tokens"],
                "output_tokens": info["completion_tokens"],
            }
            generations.append(
                [
                    Generation(
                        text=prediction["text"],
                        generation_info={"usage_metadata": usage_metadata},
                    )
                ]
            )

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "vertexai_model_garden"


def get_model(model_name: str, config_path: str, temperature: float = 0.0, **kwargs):
    """Prepares a chat model for experiments."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
    with open(config_path, "r") as read_f:
        config = json.load(read_f)

    project = config["project"]
    project_id = config["project_id"]
    if model_name in _GEMINI_MODELS:
        return ChatVertexAI(
            model_name=model_name,
            project=project,
            # location=config["models"]["gemini"]["location"],
            temperature=temperature,
            safety_settings=safety_settings,
            **kwargs,
        )
    if model_name in _GEMINI_MODELS_EXP:
        return ChatVertexAI(
            model_name=model_name,
            project=project,
            temperature=temperature,
            safety_settings=safety_settings,
            rate_limiter=rate_limiter_gemini_exp,
            **kwargs,
        )
    if model_name in ["deepseek"]:
        llm = _DeepSeekLLM(
            endpoint_id=config["models"][model_name]["endpoint_id"],
            project=project_id,
            location=config["models"][model_name]["location"],
        )
        if "max_output_tokens" in kwargs:
            return llm.bind(max_tokens=kwargs["max_output_tokens"])
        return llm
    if model_name in ["gemma_9b_it", "gemma_27b_it"]:
        llm = VertexAIModelGarden(
            endpoint_id=config["models"][model_name]["endpoint_id"],
            project=project,
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
            project=project,
            location=config["models"][model_name]["location"],
            temperature=temperature,
        )
        if "max_output_tokens" in kwargs:
            return llm.bind(max_tokens=kwargs["max_output_tokens"])
        return llm
    if model_name in ["llama_2b"]:
        llm = VertexAIModelGarden(
            endpoint_id=config["models"][model_name]["endpoint_id"],
            project=project,
            location=config["models"][model_name]["location"],
            allowed_model_args=["temperature", "max_tokens"],
        ).bind(temperature=temperature)
        if "max_output_tokens" in kwargs:
            return llm.bind(max_tokens=kwargs["max_output_tokens"])
        return llm
    if model_name in ["medllama3"]:
        llm = VertexAIModelGarden(
            endpoint_id=config["models"][model_name]["endpoint_id"],
            project=project,
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
            project=project,
            temperature=temperature,
            rate_limiter=rate_limiter_llama,
            append_tools_to_system_message=True,
        )
    if model_name == "llama3_70b":
        return get_vertex_maas_model(
            model_name="meta/llama3-70b-instruct-maas",
            project=project,
            temperature=temperature,
            rate_limiter=rate_limiter_llama,
            append_tools_to_system_message=True,
        )
    if model_name == "llama_3.2_90b":
        return get_vertex_maas_model(
            model_name="meta/llama-3.2-90b-vision-instruct-maas",
            project=project,
            temperature=temperature,
            rate_limiter=rate_limiter_llama2,
            append_tools_to_system_message=True,
        )
    if model_name == "llama_3.3_70b":
        return get_vertex_maas_model(
            model_name="meta/llama-3.3-70b-instruct-maas",
            project=project,
            temperature=temperature,
            rate_limiter=rate_limiter_llama2,
            append_tools_to_system_message=True,
        )
    if model_name in ["gpt-4", "gpt-4o"]:
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            rate_limiter=rate_limiter_gpt,
        )
    if model_name == "mistral_large":
        return get_vertex_maas_model(
            # model_name="mistral-large@2407",
            model_name="mistral-large-2411@001",
            project=project,
            temperature=temperature,
            rate_limiter=rate_limiter_mistral,
        )
    if model_name == "mistral_nemo":
        return get_vertex_maas_model(
            model_name="mistral-nemo@2407",
            project=project,
            temperature=temperature,
            rate_limiter=InMemoryRateLimiter(requests_per_second=2.0),
        )
    if model_name == "anthropic_claude":
        return ChatAnthropicVertex(
            model_name="claude-3-5-sonnet@20240620",
            project=project,
            temperature=temperature,
            rate_limiter=InMemoryRateLimiter(requests_per_second=2.0),
            location="us-east5",
        )

    if model_name == "anthropic_claude_v2":
        return ChatAnthropicVertex(
            model_name="claude-3-5-sonnet-v2@20241022",
            project=project,
            temperature=temperature,
            rate_limiter=InMemoryRateLimiter(requests_per_second=2.0),
            location="us-east5",
        )
