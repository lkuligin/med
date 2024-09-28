import threading
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class ModelGardenCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks VertexAI info."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    max_input_tokens: int = 0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"\tPrompt tokens: {self.prompt_tokens}\n"
            f"\tCompletion tokens: {self.completion_tokens}\n"
            f"Successful requests: {self.successful_requests}\n"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Runs when LLM starts running."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Runs on new LLM token. Only available when streaming is enabled."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collects token usage."""
        completion_tokens, prompt_tokens, max_tokens = 0, 0, 0
        for generations in response.generations:
            usage_metadata = None
            if len(generations) > 0:
                if generations[0].generation_info:
                    usage_metadata = generations[0].generation_info.get(
                        "usage_metadata", {}
                    )
                if not usage_metadata and generations[0].message.usage_metadata:
                    usage_metadata = generations[0].message.usage_metadata
                completion_tokens += usage_metadata.get("candidates_token_count", usage_metadata.get("output_tokens", 0))
                tokens = usage_metadata.get("prompt_token_count", usage_metadata.get("input_tokens", 0))
                prompt_tokens += tokens
                if tokens > max_tokens:
                    max_tokens = tokens

        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.successful_requests += 1
            if max_tokens > self.max_input_tokens:
                self.max_input_tokens = max_tokens



def get_callback(model_name: str):
    #if model_name in _GEMINI_MODELS:
    #    return VertexAICallbackHandler()
    
    return ModelGardenCallbackHandler()
