
import argparse
import asyncio
import csv
import json
import os
import random
import uuid

from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.runners import Runner
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_vertexai import ChatVertexAI

from _prompts import _PROMPT_SIMULATE_STEP1, _PROMPT_SIMULATE_STEP2
from _ehr import create_ehr_hint
from agents.maia_agent.agent import brca_agent

_MAX_LENGTH = 20


async def _generate_response(profile, ehr, history: str | None = None):
    
    llm = ChatVertexAI(project="kuligin-sandbox1", model="gemini-3.1-pro-preview", temperature=1.0, max_retries=100)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _PROMPT_SIMULATE_STEP1), 
            #("human", [
            #    {"type": "text", "text": "**Patient handbook:**\n"},
            #    {"type": "media", "file_uri": gcs_link, "mime_type": "application/pdf"},
            #    ]),
            ("human", _PROMPT_SIMULATE_STEP2)]
        )
    history = history if history else "START A NEW CHAT"
    response = await (prompt | llm | StrOutputParser()).ainvoke({"history": history, "profile": profile, "ehr": ehr})
    print(response)
    return response

def _merge_history(messages: list[dict]) -> str:
    history = []
    for msg in messages:
        if msg["role"] == "human":
            history.append(f"Persona A: {msg['text']}")
        else:
            history.append(f"Chatbot: {msg['text']}")
    return "\n".join(history)

async def _generate(session_service: InMemorySessionService, runner: Runner, ehr_path: str):
    suffix = ehr_path.split(".")[0]
    with open(f"profiles_{suffix}.json", "r", encoding="utf-8") as f:
        profiles = json.load(f)["profiles"]

    ehr_hint = create_ehr_hint(ehr_path)

    profile_id = random.randint(0, len(profiles) - 1)
    profile = profiles[profile_id]
    chat_history = []
    user_id = uuid.uuid4()
    session = await session_service.create_session(user_id=str(user_id), app_name="maia-test", state={"ehr": ehr_hint})
    
    while True:
        new_message = await _generate_response(profile, ehr=ehr_hint, history=_merge_history(chat_history))
        if new_message.lower().strip() == "fertig":
            break

        chat_history.append({"role": "human", "text": new_message})
        content = types.Content(role='user', parts=[types.Part(text=new_message)])

        bot_replied = False
        async for event in runner.run_async(user_id=str(user_id), session_id=session.id, new_message=content):
            if event.is_final_response() and event.content:
                final_answer = event.content.parts[0].text.strip()
                print(final_answer)
                chat_history.append({"role": "chat", "text": final_answer})
                bot_replied = True

        if not bot_replied:
            chat_history.append({"role": "chat", "text": "[no response]"})

        if len(chat_history) >= _MAX_LENGTH:
            break

    return {"chat_history": chat_history, "user_id": str(user_id), "profile_id": profile_id}


async def run(ehr_path: str):
    suffix = ehr_path.split(".")[0]
    session_service = InMemorySessionService()
    runner = Runner(agent=brca_agent, app_name="maia-test", session_service=session_service)

    simulations = []
    try:
        with open(f"simulations_{suffix}.json", "r", encoding="utf-8") as f:
            simulations = json.load(f)["simulations"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    simulation = await _generate(session_service, runner, ehr_path)
    simulations.append(simulation)
    with open(f"simulations_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({"simulations": simulations}, f, ensure_ascii=False, indent=2)
    print("simulation appended to simulations.json")

    user_id = simulation["user_id"]
    csv_file = f"{user_id}.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["simulation_id", "profile_id", "speaker", "message"])

        chat_history = simulation["chat_history"]
        simulation_id = str(user_id)
        profile_id = simulation["profile_id"]

        for message in chat_history:
            writer.writerow([simulation_id, profile_id, message["role"], message["text"]])
    print("csv generated!")

async def main(ehr_path: str):
    for _ in range(20):
        await run(ehr_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ehr_path", help="Path to the EHR JSON file")
    args = parser.parse_args()
    asyncio.run(main(args.ehr_path))
