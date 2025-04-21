from operator import itemgetter
from typing import Literal, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from qa.agents_utils import get_search_tool
from qa.chains import _PROMPT_EXTRACT_ANSWER
from qa.models import get_model
from qa.utils import _format_options, _parse_response

_REACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer the given medical exam question. Thinks step by step."
                "Always provide an argumentation for your answer, and use Google Search "
                "to support your answer."
            ),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

_PROMPT_QUESTION = ChatPromptTemplate.from_template(
    "Answer the multiple-choice exam question by picking a single correct option.\n"
    "Question: {question}.\nAnswer options:\n{options}\n."
)

_PROMPT_QUESTION_WITH_CRITIQUE = ChatPromptTemplate.from_template(
    "Answer the multiple-choice exam question by picking a single correct option.\n"
    "Question: {question}.\nAnswer options:\n{options}\n."
    "Your previous answer was:\n{answer}\n"
    "Your received the following critical feedback:\n{critique}\n"
    "Think how to adress this feedback and prepare a new answer."
)

_REFLECTION_PROMPT = ChatPromptTemplate.from_template(
    "You are a medical exam professor and you're supervising a student who is "
    "working on a medical exam with multiple-choice questions. "
    "Question: {question}.\nAnswer options:\n{options}\n."
    "The student gave the following answer:\n{answer}\n"
    "Reflect about the answer, and provide a feedback whether the answer "
    "is right or wrong. If you think the final answer is correct, reply with "
    "the final answer. Only provide critique if you think the asnwer might "
    "be incorrect. Ignore `google_search` and never invoke it."
)


def get_react_chain(model, prompt=_PROMPT_QUESTION):
    tool = get_search_tool()
    agent = create_react_agent(model, [tool], prompt=_REACT_PROMPT)
    chain = (
        {
            "question": itemgetter("question"),
            "options": RunnableLambda(lambda x: _format_options(x["options"])),
            "answer": RunnableLambda(lambda x: x.get("answer")),
            "critique": RunnableLambda(
                lambda x: x["response"].critique if x.get("response") else ""
            ),
        }
        | RunnablePassthrough.assign(messages=lambda x: [("user", prompt.format(**x))])
        | agent
    )
    return chain | RunnablePassthrough.assign(
        answer=lambda x: x["messages"][-1].content
    )


class Response(BaseModel):
    """A final response to the user."""

    answer: Optional[str] = Field(
        description="The final answer. It should be empty if critique has been provided.",
        default=None,
    )
    critique: Optional[str] = Field(
        description="A critique of the initial answer. If you think it might be incorrect, provide an acitonable feedback",
        default=None,
    )


class AgentState(TypedDict):
    question: str
    options: str
    answer: str
    critique: str
    steps: int
    response: Response


def _should_end(state: AgentState) -> Literal["react", "force_response", "parse"]:
    if state.get("response") and state["response"].answer:
        return "parse"
    if state.get("steps", 1) > 5:
        return "force_response"
    return "react"


def _get_force_generation_step(model):
    chain = _PROMPT_QUESTION_WITH_CRITIQUE | model | {"answer": lambda x: x.content}

    def _run_force_generation(state):
        parsed_state = {
            "answer": state["answer"],
            "options": state["options"],
            "critique": state["response"].critique if state.get("response") else "",
            "question": state["question"],
        }
        return chain.invoke(parsed_state)

    return _run_force_generation


def _get_reflect_step(model):
    chain = _REFLECTION_PROMPT | model.with_structured_output(Response)

    def _run(state):
        result = chain.invoke(state)
        return {"response": result, "steps": state.get("steps", 1) + 1}

    return _run


def _get_parse_step(model):
    chain = (
        _PROMPT_EXTRACT_ANSWER
        | model
        | {"answer": RunnableLambda(lambda x: _parse_response(x.content))}
    )

    def _run(state):
        if state.get("response") and state["response"].answer:
            answer_candidate = state["response"].answer
        else:
            answer_candidate = state["answer"]
        result = chain.invoke({"full_answer": answer_candidate})
        return {"response": Response(answer=result["answer"])}

    return _run


def get_workflow(model_name, max_output_tokens, temperature, config_path: str):
    model = get_model(
        model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        config_path=config_path,
    )
    model2 = get_model(model_name, temperature=0.0, config_path=config_path)
    react_agent_initial = get_react_chain(model)
    react_agent = get_react_chain(model, prompt=_PROMPT_QUESTION_WITH_CRITIQUE)

    def _run_react_agent_initial(state):
        result = react_agent_initial.invoke(state)
        return {"answer": result["answer"]}

    def _run_react_agent(state):
        result = react_agent.invoke(state)
        return {"answer": result["answer"]}

    force_response_step = _get_force_generation_step(model2)
    reflect_step = _get_reflect_step(model)
    parse_step = _get_parse_step(model2)

    workflow = StateGraph(AgentState)
    workflow.add_node("react_start", _run_react_agent_initial)
    workflow.add_node("react", _run_react_agent)
    workflow.add_node("reflect", reflect_step)
    workflow.add_node("force_response", force_response_step)
    workflow.add_node("parse", parse_step)

    workflow.add_edge(START, "react_start")
    workflow.add_edge("react_start", "reflect")
    workflow.add_edge("react", "reflect")
    workflow.add_conditional_edges("reflect", _should_end)
    workflow.add_edge("force_response", "parse")
    workflow.add_edge("parse", END)
    return workflow.compile()


def get_reflection_chain(
    model_name, max_output_tokens=2048, temperature=0.0, config_path=None
):
    agent = get_workflow(
        model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        config_path=config_path,
    )
    return agent | {"answer": lambda x: x["response"].answer}
