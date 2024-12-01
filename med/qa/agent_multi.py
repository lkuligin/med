from operator import itemgetter, add
from typing import Annotated, List, Literal, Optional, Tuple, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from qa.agents_utils import get_search_tool
from qa.chains import _PROMPT_EXTRACT_ANSWER
from qa.models import get_model
from qa.utils import _format_options, _parse_response
from qa.agent_reflect import _REACT_PROMPT, _PROMPT_QUESTION


_STUDENT_PROMPT = ChatPromptTemplate.from_template(
    "STUDENT is a very good and hard working student who studies medicine and is "
    "curious to learn new things. THeir task is to find an answer to the "
    "multiple-choice question and pick a single correct option.\n"
    "Question: {question}.\nAnswer options:\n{options}\n."
    "They have been researching the answer and working with the professor."
    "Below is a history of a conversation between a STUDENT and their PROFESSOR. "
    "\n\n{history}\nWrite the next STUDENT's response to the PROFESSOR's critiqiue."
)

_PROFESSOR_PROMPT = ChatPromptTemplate.from_template(
    "You are an experienced medical professor with very deep knowledge of medicine. "
    "You're helping a student to find a single correct option to answer a "
    "multiple-choice questions. Question: {question}.\nAnswer options:\n{options}\n."
    "This your previous conversation with the student:\n{history}\n"
    " Think whether student's reasoning is correct and reply to the student. "
    "If it is not correct, provide a students with a positive critique that would "
    "help them to research and find the correct answer. Never reveal the answer "
    "itself. Stay positive and ensure the best learning experience for your student."
    "Now reply back to the student."
)

_PROMPT_QUESTION_WITH_CRITIQUE = ChatPromptTemplate.from_template(
    "Answer the multiple-choice exam question by picking a single correct option.\n"
    "Question: {question}.\nAnswer options:\n{options}\n."
    "This is the conversation between ths student and their professor "
    "debating this question:\n{history}\n"
    "Based on this conversation, generate the final answer."
)


class Response(BaseModel):
    """A response from the professor."""

    is_answer_correct: bool = Field(
        description="Whether the student's answer is correct",
        default=False,
    )
    professor_reply: Optional[str] = Field(
        description="A reply from the professor to the conversation with the student.",
        default=None,
    )


class AgentState(TypedDict):
    question: str
    options: str
    scratchpad: Annotated[List[Tuple[str, str]], add]
    ready: bool
    final_answer: str


def _should_end(state: AgentState) -> Literal["student", "force_response", "parse"]:
    if state.get("ready"):
        return "parse"
    if len(state["scratchpad"]) > 10:
        return "force_response"
    return "student"


def _format_history(state):
    return "\n".join([f"{role}: {msg}" for role, msg in state["scratchpad"]])


def get_react_chain(model, prompt=_PROMPT_QUESTION):
    tool = get_search_tool()
    agent = create_react_agent(model, [tool], messages_modifier=_REACT_PROMPT)
    chain = (
        {
            "question": itemgetter("question"),
            "history": RunnableLambda(_format_history),
            "options": itemgetter("options"),
        }
        | RunnableLambda(lambda x: {"messages": [("user", prompt.format(**x))]})
        | agent
        | RunnableLambda(lambda x: x["messages"][-1].content)
    )
    return chain


def _get_run_professor_step(model):
    chain = _PROFESSOR_PROMPT | model.with_structured_output(Response)

    def _run(state):
        tool = get_search_tool()
        scratchpad = state["scratchpad"]
        history = "\n".join([f"{role}: {msg}" for role, msg in scratchpad])
        result = chain.invoke(
            {
                "options": state["options"],
                "history": history,
                "question": state["question"],
            }
        )
        return {
            "scratchpad": [("PROFESSOR", result.professor_reply)],
            "ready": result.is_answer_correct,
        }

    return _run


def _get_force_generation_step(model):
    chain = _PROMPT_QUESTION_WITH_CRITIQUE | model | StrOutputParser()

    def _run_force_generation(state):
        parsed_state = {
            "options": state["options"],
            "history": RunnableLambda(_format_history),
            "question": state["question"],
        }
        result = chain.invoke(parsed_state)
        return {"scratchpad": [("STUDENT", result)], "ready": False}

    return _run_force_generation


def _get_parse_step(model):
    chain = (
        _PROMPT_EXTRACT_ANSWER
        | model
        | {"answer": RunnableLambda(lambda x: _parse_response(x.content))}
    )

    def _run(state):
        is_ready = state["ready"]
        if is_ready:
            answer_candidate = state["scratchpad"][-2][1]
        else:
            answer_candidate = state["scratchpad"][-1][1]
        result = chain.invoke({"full_answer": answer_candidate})
        return {"final_answer": result["answer"]}

    return _run


def get_workflow(
    model_name: str, config_path: str, max_output_tokens: int, temperature: float
):
    model = get_model(
        model_name=model_name,
        config_path=config_path,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    model2 = get_model(model_name=model_name, config_path=config_path, temperature=0.0)
    react_agent_initial = get_react_chain(model)
    react_agent = get_react_chain(model, prompt=_STUDENT_PROMPT)

    def _run_react_agent_initial(state):
        formatted_options = _format_options(state["options"])
        parsed_state = {**state}
        parsed_state["options"] = formatted_options
        result = react_agent_initial.invoke(parsed_state)
        return {"scratchpad": [("STUDENT", result)], "options": formatted_options}

    def _run_react_agent(state):
        result = react_agent.invoke(state)
        return {"scratchpad": [("STUDENT", result)]}

    force_response_step = _get_force_generation_step(model2)
    _run_professor = _get_run_professor_step(model)
    parse_step = _get_parse_step(model2)

    workflow = StateGraph(AgentState)
    workflow.add_node("initial", _run_react_agent_initial)
    workflow.add_node("student", _run_react_agent)
    workflow.add_node("professor", _run_professor)
    workflow.add_node("force_response", force_response_step)
    workflow.add_node("parse", parse_step)

    workflow.add_edge(START, "initial")
    workflow.add_edge("initial", "professor")
    workflow.add_edge("student", "professor")
    workflow.add_conditional_edges("professor", _should_end)
    workflow.add_edge("force_response", "parse")
    workflow.add_edge("parse", END)
    return workflow.compile()


def get_diag_chain(
    model_name: str,
    config_path: str,
    max_output_tokens: int = 2048,
    temperature: float = 0.0,
):
    agent = get_workflow(
        model_name=model_name,
        config_path=config_path,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return agent | {"answer": lambda x: x["final_answer"]}
