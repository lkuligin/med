from collections import Counter
from operator import itemgetter
from typing import Any, Dict

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.prebuilt import create_react_agent

from qa.agent_plan import get_workflow
from qa.agents_utils import get_search_tool
from qa.models import get_model
from qa.utils import _format_options, _parse_response

_PROMPT_COT = (
    "You're taking an exam with a multiple-choice question. The question is:\n"
    "{question}.\n\nThe answer choices are:\n{options}\n."
    "Think step-by-step:\n\nAnalyze the question: What is the main topic or "
    "concept being tested? Are there any keywords or clues?\n"
    "Consider each answer choice: Does the answer choice make sense based on "
    "the information given in the question? Is there any evidence to support "
    "or refuse the choice?\nEliminate incorrect choices: Can you rule out any "
    "answer choices based on your analysis?\nEvaluate the remaining choices: "
    "Which answer choice is the most likely to be correct? Is there any reason "
    "to doubt the accuracy of your choice? Once you've completed your analysis, "
    "provide your answer and explain your reasoning."
)

_PROMPT_CRITIQUE = (
    "You are knowledgeable professor of medicine. A student is answering an exam "
    "question. The question was:\n{question}."
    "The student provided the following answer with the reason"
    "Criticize their reasoning and highlight potential flaws and errors."
    "\nSTUDENT's ANSWER:\n:{cot_answer}."
)

_PROMPT_EXTRACT_ANSWER = PromptTemplate.from_template(
    "You're given a full answer for a multiple-choice question. The intermediate answer "
    "includes reasoning and explanation.\n FULL ANSWER WITH REASONING:\n{full_answer}\n"
    "Your task is to extract only the final "
    "answer itself. Answer only a single letter and don't give any explanations: "
)


class Sampler:
    def __init__(self, chain, max_retries: int = 10):
        self._chain = chain
        self._max_retries = max_retries

    def run(self, entry, n=10):
        results = [self._run(entry) for _ in range(n)]
        counter = Counter(results)
        winner, _ = counter.most_common()[0]
        return winner

    def run_batch(self, entries):
        results = self._chain.batch(entries, config={"max_concurrency": 5})
        return results

    def _run(self, entry, retry: int = 0):
        if retry > self._max_retries:
            raise ValueError("Max retries reached")
        try:
            return self._chain.invoke(entry)
        except Exception:
            return self._run(entry, retry + 1)


def _parse_response_llama(response: BaseMessage) -> str:
    answer = response.strip("”.:[\"'(\\ *\n `")[0].upper()
    return answer


def get_simple_chain(model_name: str, sample_size: int = 10, temperature: float = 0.0):
    llm = get_model(model_name, temperature=temperature)

    prompt = PromptTemplate.from_template(
        (
            "Choose the right answer to the following question:\n{question}\n"
            "You need to choose the correct options out of:\n{options}\n"
            "Answer only a single letter and don't give any explanations:"
        )
    )

    if model_name in ["llama_2b", "medllama3", "gemma_9b_it", "gemma_27b_it"]:
        chain = (
            {
                "question": itemgetter("question"),
                "options": RunnableLambda(lambda x: _format_options(x["options"])),
            }
            | prompt
            | llm
            | {
                "full_output": RunnableLambda(lambda x: x),
                "answer": RunnableLambda(_parse_response_llama),
            }
        )

        return chain

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


def get_cot_chain(model_name: str, max_output_tokens: int = 2048):
    llm = get_model(model_name, temperature=0.0, max_output_tokens=max_output_tokens)
    if model_name in ["llama_2b", "gemma_2b"]:
        llm2 = get_model(model_name, temperature=0.0)
    else:
        llm2 = llm

    prompt_first = PromptTemplate.from_template(_PROMPT_COT)

    chain_start = (
        {
            "question": itemgetter("question"),
            "options": RunnableLambda(lambda x: _format_options(x["options"])),
        }
        | RunnablePassthrough.assign(first_step=(prompt_first | llm).invoke)
        | RunnablePassthrough.assign(
            final_answer=lambda x: (
                (_PROMPT_EXTRACT_ANSWER | llm2).invoke(x["first_step"])
            )
        )
    )
    if model_name == "llama_2b":
        chain = chain_start | {
            "full_output": RunnableLambda(lambda x: x["first_step"]),
            "answer": RunnableLambda(
                lambda x: _parse_response_llama(x["final_answer"])
            ),
        }
        return chain
    parser = StrOutputParser()
    chain = chain_start | {
        "full_output": RunnableLambda(lambda x: parser.invoke(input=x["first_step"])),
        "answer": RunnableLambda(lambda x: _parse_response(x["final_answer"])),
    }

    return chain


def get_refl_chain(model_name: str, max_output_tokens: int = 2048):
    llm = get_model(model_name, temperature=0.0, max_output_tokens=max_output_tokens)
    if model_name in ["llama_2b", "gemma_2b"]:
        llm2 = get_model(model_name, temperature=0.0)
    else:
        llm2 = llm

    prompt_cot = PromptTemplate.from_template(_PROMPT_COT)
    prompt_critique = PromptTemplate.from_template(_PROMPT_CRITIQUE)

    prompt_fin = PromptTemplate.from_template(
        "You're taking an exam with a multiple-choice question. The question is:\n"
        "{question}.\n\nThe answer choices are:\n{options}\n."
        "YOUR ANSWER:\n{cot_answer}\n\n"
        "You received the following critique for this answer:\n{critique}\n\n"
        "Now take this critique into account and finalize your answer."
    )

    chain = (
        {
            "question": itemgetter("question"),
            "options": RunnableLambda(lambda x: _format_options(x["options"])),
        }
        | RunnablePassthrough.assign(cot_answer=(prompt_cot | llm))
        | RunnablePassthrough.assign(critique=(prompt_critique | llm))
        | RunnablePassthrough.assign(full_answer=(prompt_fin | llm))
        | RunnablePassthrough.assign(final_answer=(_PROMPT_EXTRACT_ANSWER | llm2))
        | {
            "final_answer": RunnableLambda(lambda x: x["final_answer"]),
            "answer": RunnableLambda(lambda x: _parse_response(x["final_answer"])),
        }
    )

    return chain


def get_react_chain(model_name: str, max_output_tokens: int = 2048):
    llm = get_model(model_name, temperature=0.0, max_output_tokens=max_output_tokens)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Try to answer the given medical exam question. Thinks step by step "
                    " and get required information from Google Search if needed."
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    tool = get_search_tool()
    agent = create_react_agent(llm, [tool], messages_modifier=prompt)
    chain = (
        {
            "question": itemgetter("question"),
            "options": RunnableLambda(lambda x: _format_options(x["options"])),
        }
        | RunnablePassthrough.assign(
            messages=lambda x: [("user", _PROMPT_COT.format(**x))]
        )
        | agent
        | RunnablePassthrough.assign(
            final_answer=lambda x: (
                (_PROMPT_EXTRACT_ANSWER | llm).invoke(x["messages"][-1])
            )
        )
        | {"answer": RunnableLambda(lambda x: _parse_response(x["final_answer"]))}
    )
    return chain


def get_plan_chain(model_name: str, max_output_tokens: int = 2048):
    llm = get_model(model_name, temperature=0.0, max_output_tokens=max_output_tokens)
    agent = get_workflow(llm)

    chain = (
        {
            "question": itemgetter("question"),
            "options": RunnableLambda(lambda x: _format_options(x["options"])),
        }
        | agent
        | RunnablePassthrough.assign(
            final_answer=lambda x: (
                (_PROMPT_EXTRACT_ANSWER | llm).invoke(x["response"])
            )
        )
        | {"answer": RunnableLambda(lambda x: _parse_response(x["final_answer"]))}
    )
    return chain


def _format_question(entry: Dict[str, Any]) -> str:
    prompt_qa = PromptTemplate.from_template(
        (
            "Choose the right answer to the following question:\n{question}\n"
            "You need to choose the correct options out of:\n{options}\n"
            "Answer only a single letter and don't give any explanations:"
        )
    )
    return prompt_qa.format(
        question=entry["question"], options=_format_options(entry["options"])
    )
