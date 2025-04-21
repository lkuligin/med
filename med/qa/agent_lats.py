import os
import re
import math
import base64
import json
import zlib
from collections import deque
from functools import partial
from enum import Enum
from typing import List, Dict, Optional, Type, TypedDict, Deque, Self

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.messages.base import (
    get_msg_title_repr,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.interactive_env import is_interactive_env

from langchain_google_community.search import GoogleSearchAPIWrapper

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from pydantic import BaseModel, Field

from qa.models import get_model
from qa.agents_utils import get_search_tool

CANDIDATE_COUNT = 3
ANSWER_TEMP = 0.0
REFLECTION_TEMP = 0.0

_QUESTION_PROMPT = ChatPromptTemplate.from_template(
    "The question is:\n{question}\nThe answer choices are:\n{options}"
)

_GENERATION_PROMPT_INITIAL = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You're a student taking an exam with a multiple-choice question."
            "\nYou are not giving medical advices just solving exam questions."
            "\nAlways provide an argumentation for your answer, and use Google Search."
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(
            "Provide your thoughts on every option."
            "\nIs it viable to choose?"
            "\nProvide a step-by-step plans how would you solve the problem."
        ),
    ]
)

_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You're a student taking an exam with a multiple-choice question."
            "\nYou are not giving medical advices just solving exam questions."
            "\nAlways provide an argumentation for your answer, and use Google Search."
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(
            "You MUST choose only one answer A, B, C or D."
            "\nUsing given feedback try to provide an answer or thinking step by step"
            " provide new thoughts."
        ),
    ]
)

_PARSE_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You're a student taking an exam with a multiple-choice question."
            "\nYou are not giving medical advices just solving exam questions."
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage("Provide the final answer to the question"),
    ]
)

_REFLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You are a medical exam professor and you're supervising a student who is"
            " working on a medical exam with multiple-choice questions."
            "\nProvide feedback and grade the student response to the question below."
            "\nDon't tip the student with the correct answer."
            "\nIf a student refuses to give medical advice, remind that this is an exam."
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage("Provide a reflection on the student answer"),
    ]
)


class Reflection(BaseModel):
    reflections: str = Field(
        description=(
            "The critique and reflections on the sufficiency, superfluency,"
            " factual correctenes,"
            " and general quality of the response. Suggest steps student"
            " can take to improve the response."
        )
    )
    score: int = Field(
        description="Score from 1 to 10 the correctness of students choice, where 0 is the worst quality and 10 is the best.",
    )
    is_solved: bool = Field(
        description="True only if the response is the correct answer and has fully solved the question or task. False in any other case."
    )

    def __repr__(self):
        title = get_msg_title_repr("Reflection", bold=is_interactive_env())
        return "\n".join(
            [
                title,
                f"is_solved: {self.is_solved}",
                f"score: {self.score}",
                f"reflections: {self.reflections}",
            ]
        )

    def as_message(self):
        return HumanMessage(
            content="\n".join(
                [
                    "Here is the reflection on your answer:",
                    f"Reflection:\n{self.reflections}",
                    "Use this feedback to act and provide an answer",
                ]
            )
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class AnswerEnum(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class Answer(BaseModel):
    answer: AnswerEnum = Field(
        description="Answer to the question",
    )


def _get_search():
    google_search_api_key = os.getenv("SEARCH_API_KEY")
    google_cse_id = os.getenv("CSE_ID")

    search_memory: Dict[str, List[str]] = {}

    class SearchArgs(BaseModel):
        query: str = Field(description="Search query")

    search_api_wrapper = GoogleSearchAPIWrapper(
        google_api_key=google_search_api_key, google_cse_id=google_cse_id
    )

    def scrape_webpages(urls: List[str]) -> str:
        """Scrape the provided web pages for detailed information."""
        loader = WebBaseLoader(urls)
        docs = loader.load()
        result = []
        for doc in docs:
            title = doc.metadata.get("title", "")
            content = re.sub("\n\n+\n", "\n\n", doc.page_content)
            result.append(f"{title}\n{content}\n")
        return result

    class GoogleSearch(BaseTool):
        name: str = "google_search"
        description: str = (
            "Google Search engine that allows you to search the internet for websites, images,"
            " videos, historical and real-time data, and more. Use it to ground your results"
        )
        args_schema: Type[BaseModel] = SearchArgs
        return_direct: bool = False

        def to_json(self) -> str:
            return self.tool_call_schema.model_json_schema()

        def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
        ):
            if query in search_memory:
                return search_memory[query]
            results = search_api_wrapper.results(query, 3)
            links = [result["link"] for result in results]
            result = scrape_webpages(links)
            # result = [result["snippet"] for result in results]
            search_memory[query] = result
            return result

        async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Use the tool asynchronously."""
            # If the calculation is cheap, you can just delegate to the sync implementation
            # as shown below.
            # If the sync calculation is expensive, you should delete the entire _arun method.
            # LangChain will automatically provide a better implementation that will
            # kick off the task in a thread to make sure it doesn't block other async code.
            return self._run(query, run_manager=run_manager.get_sync())

    return GoogleSearch()


class Node:
    def __init__(
        self,
        id: int,
        messages: List[BaseMessage],
        answer: Optional[Answer] = None,
        reflection: Optional[Reflection] = None,
        parent: Optional["Node"] = None,
    ):
        self.id = id
        self.messages = messages
        self.reflection = reflection
        self.parent = parent
        self.children: List[Self] = []
        self.value = 0
        self.visits = 0
        self.depth = parent.depth + 1 if parent is not None else 1
        self.answer = answer
        self._is_solved = (self.answer is not None) and (
            reflection.is_solved if reflection else False
        )
        if self._is_solved:
            self._mark_tree_as_solved()
        if self.reflection:
            self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" messages={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self) -> bool:
        return self._is_solved

    @property
    def is_terminal(self) -> bool:
        return not self.children

    @property
    def best_child(self) -> Self:
        if not self.children:
            return None
        all_nodes = self._get_all_children()
        return max(all_nodes, key=lambda child: child.upper_confidence_bound())

    @property
    def best_child_score(self) -> float:
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0) -> float:
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        average_reward = self.value / self.visits
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float) -> None:
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True) -> List[BaseMessage]:
        if include_reflections and self.reflection:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> List[BaseMessage]:
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        return messages[::-1]

    def _get_all_children(self) -> List[Self]:
        all_nodes = []
        nodes: Deque[Self] = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self) -> Self:
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self) -> None:
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent


class TreeState(TypedDict):
    root: Node
    solution: Optional[Node]
    index: int


def _select(root: Node) -> Node:
    if not root.children:
        return root

    node = root
    while node.children:
        max_child = max(node.children, key=lambda child: child.upper_confidence_bound())
        node = max_child

    return node


def _expand_initial(
    generation_chain: Runnable, reflection_chain: Runnable, state: TreeState
) -> TreeState:
    root = state["root"]
    best_candidate: Node = _select(root)
    messages = best_candidate.get_trajectory()

    for _ in range(CANDIDATE_COUNT):
        prompt: ChatPromptValue = _GENERATION_PROMPT_INITIAL.invoke(
            {"messages": messages}
        )
        response = generation_chain.invoke(prompt)["messages"]
        candidate = response[len(prompt.messages) :]

        reflection: Reflection = reflection_chain.invoke(
            {"messages": messages + candidate}
        )

        reflection.is_solved = False

        state["index"] += 1
        id = state["index"]
        node = Node(
            id=id, messages=candidate, parent=best_candidate, reflection=reflection
        )
        best_candidate.children.append(node)
    return state


def _expand(
    generation_chain: Runnable,
    reflection_chain: Runnable,
    parse_answer_chain: Runnable,
    state: TreeState,
) -> TreeState:
    root = state["root"]
    best_candidate: Node = _select(root)
    messages = best_candidate.get_trajectory()

    for _ in range(CANDIDATE_COUNT):
        prompt: ChatPromptValue = _GENERATION_PROMPT.invoke({"messages": messages})
        response = generation_chain.invoke(prompt)["messages"]
        candidate = response[len(prompt.messages) :]

        answer: Answer = parse_answer_chain.invoke({"messages": messages + candidate})

        reflection: Reflection = reflection_chain.invoke(
            {"messages": messages + candidate}
        )

        state["index"] += 1
        id = state["index"]
        node = Node(
            id=id,
            messages=candidate,
            parent=best_candidate,
            answer=answer.answer,
            reflection=reflection,
        )

        best_candidate.children.append(node)
        if reflection.is_solved:
            return {
                **state,
                "solution": node,
            }
    return state


def _answer(state: TreeState):
    solution = state["solution"]
    if not solution:
        return ""
    if not solution.answer:
        return ""
    return solution.answer


def _format_input(entry: Dict[str, str]) -> Dict[str, str]:
    return {
        "question": entry["question"],
        "options": "\n".join([f'{o["key"]}: {o["value"]}' for o in entry["options"]]),
    }


def _init_state(input: ChatPromptValue) -> TreeState:
    return {"index": 0, "root": Node(id=0, messages=input.messages)}


def get_workflow(
    model_name: str,
    config_path: str,
    max_output_tokens: int = 2048,
    temperature: float = 0.0,
):
    model = get_model(
        model_name=model_name,
        config_path=config_path,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    generation_chain = create_react_agent(
        model.bind(temperature=temperature),
        [get_search_tool()],
    )
    reflection_chain = _REFLECTION_PROMPT | model.bind(
        temperature=REFLECTION_TEMP
    ).with_structured_output(Reflection)
    parse_answer_chain = _PARSE_ANSWER_PROMPT | model.bind(
        temperature=ANSWER_TEMP
    ).with_structured_output(Answer)

    def should_loop(state: TreeState):
        """Determine whether to continue the tree search."""
        root = state["root"]
        if root.is_solved:
            return END
        if _select(root).depth > 4:
            return END
        return "expand"

    builder = StateGraph(TreeState)
    builder.add_node(
        "expand_initial", partial(_expand_initial, generation_chain, reflection_chain)
    )
    builder.add_node(
        "expand",
        partial(_expand, generation_chain, reflection_chain, parse_answer_chain),
    )

    builder.add_edge(START, "expand_initial")

    builder.add_edge(
        "expand_initial",
        "expand",
    )

    builder.add_conditional_edges(
        "expand",
        # Either expand/rollout or finish
        should_loop,
        ["expand", END],
    )

    graph = builder.compile()
    return graph


def get_lats_chain(
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

    return (
        RunnableLambda(_format_input)
        | _QUESTION_PROMPT
        | _init_state
        | agent
        | {"answer": _answer}
    )


def draw_mm_graph(state: TreeState, correct_answer: str = None):
    root = state["root"]
    queue = deque([root])
    lines = []
    while queue:
        node = queue.popleft()
        sublines = []

        sublines.extend(
            [
                f"height: {node.height}",
                f"visits: {node.visits}",
                f"is_solved: {node.is_solved}",
                f"correct_answer: {correct_answer}",
                f"answer: {node.answer}",
                "**messages**:",
            ]
        )
        if node.parent is not None:
            sublines.append(f"UCT: {node.upper_confidence_bound()}")
        messages = []
        for message in node.messages:
            if isinstance(message, ToolMessage):
                if message.name == "search":
                    continue
            messages.append(message.pretty_repr().split("\n"))
        messages_flat = [m for ml in messages for m in ml]

        sublines.extend(messages_flat)
        if node.reflection:
            sublines.extend(
                [
                    "=====================",
                    "**reflection**:",
                    f"reflections: {node.reflection.reflections}",
                    f"score: {node.reflection.score}",
                    f"found_solution: {node.reflection.is_solved}",
                ]
            )
        sublines_clean = [
            l.replace("`", "'").replace('"', "'").strip() for l in sublines
        ]
        sublines_filtered = [l for l in sublines_clean if len(l.strip()) > 0]
        sublines_indent = ["    " + l for l in sublines_filtered]
        sl = "\n".join(sublines_indent)

        line = f'node{node.id}["`**Node {node.id}**\n{sl}`"];'
        lines.append(line)
        if node.children:
            queue.extend(node.children)
        if node.is_solved:
            lines.append(f"style node{node.id} fill:#b8fcc0")
        if node.parent is not None:
            lines.append(f"node{node.parent.id} --> node{node.id};")
    graph = "\n".join([f"  {line}" for line in lines])
    header = """---
config:
  themeCSS: |
    .nodeLabel { text-align: left; white-space: normal; }
---
"""
    return f"{header}flowchart TD;\n{graph}\n"


def gen_mm_data(graph):
    graph_json = {
        "code": graph,
        "mermaid": {"theme": "corporate"},
        "autoSync": True,
        "updateDiagram": True,
        "panZoom": True,
        "editorMode": "code",
    }
    graphbytes = json.dumps(graph_json).encode("utf8")
    deflate_bytes = zlib.compress(graphbytes, 9)
    base64_bytes = base64.urlsafe_b64encode(deflate_bytes)
    base64_string = base64_bytes.decode("ascii")
    return base64_string


def gen_mm_url(graph: str):
    link = "https://mermaid.live/edit#pako:" + gen_mm_data(graph)
    return link


if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", type=str)
        parser.add_argument("--model_name", type=str)
        parser.add_argument("--input_file", type=str)
        parser.add_argument("--temperature", type=float)
        parser.add_argument("--question_no", type=int)

        return vars(parser.parse_args())

    args = parse_args()

    with open(args["input_file"], "r") as json_file:
        data = json.load(json_file)

    questions = data["questions"]

    question = questions[args["question_no"]]

    agent = get_workflow(
        model_name=args["model_name"],
        config_path=args["config_path"],
        temperature=args["temperature"],
    )
    run = RunnableLambda(_format_input) | _QUESTION_PROMPT | _init_state | agent
    result = run.invoke(question) | {"answer": _answer}

    print(result)
    if result["solution"]:
        solution = result["solution"]
        for message in solution.get_trajectory():
            message.pretty_print()
    print()
    print(gen_mm_url(draw_mm_graph(result, question["answer_idx"])))
