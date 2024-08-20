from collections import Counter
from typing import Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from operator import itemgetter

from qa.models import get_model, _GEMINI_MODELS


class Sampler:
    def __init__(self, chain, max_retries: int = 10):
        self._chain = chain
        self._max_retries = max_retries

    def run(self, entry, n=10):
        results = [self._run(entry) for _ in range(n)]
        counter = Counter(results)
        winner, _ = counter.most_common()[0]
        return winner

    def _run(self, entry, retry: int = 0):
        if retry > self._max_retries:
            raise ValueError("Max retries reached")
        try:
            return chain.invoke(entry)
        except Exception as e:
            return self._run(entry, retry + 1)


def _format_options(entry: Dict[str, str]) -> str:
    return "\n".join([f"{value['key']}: {value['value']}" for value in entry])


def _parse_response(response: BaseMessage) -> str:
    return response.content.strip()[0].upper()


def get_chain(model_name: str, chain_type: str, sample_size: int = 1):
    match chain_type:
        case "simple":
            return get_simple_chain(model_name)


def get_simple_chain(model_name: str, sample_size: int = 10):
    llm = get_model(model_name)

    prompt = PromptTemplate.from_template(
        (
            "Choose the right answer to the following question:\n{question}\n."
            "You need to choose the correct options out of:\n{options}\n."
            "Answer only a single letter and don't give any explanations:"
        )
    )

    if model_name in _GEMINI_MODELS:
        chain = (
            {
                "question": itemgetter("question"),
                "options": RunnableLambda(lambda x: _format_options(x["options"])),
            }
            | prompt
            | llm
            | {
                "full_output": RunnableLambda(lambda x: x.content),
                "answer": RunnableLambda(_parse_response),
            }
        )

    return chain
