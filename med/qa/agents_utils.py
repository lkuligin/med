import os
from typing import List, Optional

from langchain_core.messages import ToolMessage
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun
from pydantic import BaseModel, Field

google_search_api_key = os.getenv("SEARCH_API_KEY")
google_cse_id = os.getenv("CSE_ID")

_PROMPT_COT = (
    "You're taking an exam with a multiple-choice question. The question is:\n"
    "{question}.\n\nThe answer choices are:\n{options}\n."
    "Think step-by-step:\n\nAnalyze the question: What is the main topic or "
    "concept being tested? Are there any keywords or clues?\n"
    "Consider each answer choice: Does the answer choice make sense based on "
    "the information given in the question? Is there any evidence to support "
    "or refute the choice?\nEliminate incorrect choices: Can you rule out any "
    "answer choices based on your analysis?\nEvaluate the remaining choices: "
    "Which answer choice is the most likely to be correct? Is there any reason "
    "to doubt the accuracy of your choice? Once you've completed your analysis, "
    "provide your answer and explain your reasoning."
)


class Queries(BaseModel):
    """List of search queries to gather information to answer the exam question"""

    search_queries: List[str] = Field(
        description="Search queris to gather additional information and improve your answer.",
    )


class StructuredRunnableWithRetries:
    def __init__(
        self,
        runnable,
        validator,
        max_retries: int = 3,
        answer_key: Optional[str] = None,
    ):
        self._runnable = runnable
        self._validator = validator
        self._max_retries = max_retries
        self._answer_key = answer_key

    def run(self, state: list):
        for attempt in range(self._max_retries):
            response = self._runnable.invoke(
                {"messages": state}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                response_parsed = self._validator.invoke(response)
                if self._answer_key:
                    return {self._answer_key: response_parsed[0]}
                return response

            except ValidationError as e:
                if not response.tool_calls:
                    err = "No tool cals found! Please, make sure to invoke a tool"
                    tool_call_id = None
                else:
                    err = (
                        f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors."
                    )
                    tool_call_id = response.tool_calls[0]["id"]
                state = state + [
                    response,
                    ToolMessage(content=err, tool_call_id=tool_call_id),
                ]
        return []


def get_search_info(search_queries: list[str], k: int = 5) -> str:
    """Prepares context based on generated search queries."""
    results = []
    search = GoogleSearchAPIWrapper(
        k=5, google_api_key=google_search_api_key, google_cse_id=google_cse_id
    )
    for query in search_queries:
        search_result = ""
        for result in search.results(query, num_results=3):
            if "title" in result and "snippet" in result:
                search_result += (
                    f'Source: {result["title"]}.\nShort snippet: {result["snippet"]}\n'
                )
        if search_result:
            results.append(search_result)
    output = ""
    for query, result in zip(search_queries, results):
        output += f"Search query: '{query}'. Results:\n{results}\n"
    return output


def get_search_tool(n: int = 3):
    search = GoogleSearchAPIWrapper(
        k=n, google_api_key=google_search_api_key, google_cse_id=google_cse_id
    )
    return GoogleSearchRun(api_wrapper=search)
