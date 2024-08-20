from enum import Enum
import json

from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold


_GEMINI_MODELS = [
    "gemini-1.5-pro-001",
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-1.5-flash-001",
]


def get_model(model_name: str):
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
            temperature=0.0,
            safety_settings=safety_settings,
        )
    if model_name == "gemma-2b":
        return GemmaChatVertexAIModelGarden(
            endpoint_id=config["models"]["gemma_2b"]["endpoint_id"],
            project="kuligin-sandbox1",
            location=config["models"]["gemma_2b"]["location"],
        )
