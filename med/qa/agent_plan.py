from typing import Annotated, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
)
from langgraph.graph import StateGraph, START, END

import operator

from qa.utils import _parse_response
from qa.agents_utils import Queries, get_search_info

class AgentState(TypedDict):
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    options: str
    context: Annotated[str, operator.add]
    queries: Queries
    steps: int
    response: str


class Act(BaseModel):
    """Action to perform."""

    queries: Optional[Queries] = Field(
        description="Queries to run across Google Search to gather additional information. Either queries or response should be set.",
        default=None
    )
    response: Optional[str] = Field(
        description="A final response to the question. If you need to gather additional information from Google Search, it should be None.",
        default=None
    )

def _run_step(state):
    return {
        "context": get_search_info(state["queries"].search_queries),
        "steps": state.get("steps", 1) + 1
    }

def _should_end(state: AgentState) -> Literal["run", "force_response", END]:
    if "response" in state and state["response"]:
        return END
    if state.get("steps", 1) > 5 or len(state.get("context")) > 80000:
        return "force_response"
    return "run"


def get_force_generation(model):
    generation_prompt_template = (
        "You're taking a medical exam with a multiple-choice question. The question is:\n"
        "{question}.\n\n You have gathered the following information from Google Search:\n"
        "{context}\n Think step by step and answer the question given answer "
        "choices are:\n{options}\n."
    )
    generation_prompt = ChatPromptTemplate.from_template(generation_prompt_template)
    return generation_prompt | model | {
        "response": RunnableLambda(lambda x: Act(response=_parse_response(x.content))),
    }


def get_planner(model):
    
    planner_prompt_template = (
        "You're taking a mdecial exam with a multiple-choice question. The question is:\n"
        "{question}.\n\nThe answer choices are:\n{options}\n "
        "You have gather the following information from Google Search:\n"
        "{context}\n Think whether you have enough data to answer the exam question. If you "
        "need more information from Google Search, generate a list of additional "
        "search queries to run. Otherwise, prepare the final answer in the response field."
        "Do not use search too much, try to answer the question when you have enough information."
    )
    planner_prompt = ChatPromptTemplate.from_template(planner_prompt_template)
    return planner_prompt | model.with_structured_output(Act) | {"queries": RunnableLambda(lambda x: Queries(search_queries=x.queries.search_queries[:5])), "response": RunnableLambda(lambda x: x.response)}


def get_workflow(model):
    planner = get_planner(model)
    generation = get_force_generation(model)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("plan", planner.invoke)
    workflow.add_node("run", _run_step)
    workflow.add_node("force_response", generation)
    workflow.add_edge(START, "plan")
    workflow.add_edge("run", "plan")
    workflow.add_conditional_edges("plan", _should_end)
    return workflow.compile()

