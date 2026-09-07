import argparse
import asyncio
import csv
import json
import os
import random
import re
import uuid
from typing import Any

import httpx
from google import genai
from google.genai import types

from _ehr import create_ehr_hint
from _prompts import _PROMPT_SIMULATE_STEP1, _PROMPT_SIMULATE_STEP2

_MAX_LENGTH = 1000
_REPLICA_THRESHOLD = 300
_DEFAULT_MODEL = "gemini-3.8-flash"
_MAX_TURNS = 50


def _get_ehr_suffix(ehr_path: str) -> str:
    """Extracts a clean filename suffix from the EHR file path."""
    filename = os.path.basename(ehr_path)
    suffix = os.path.splitext(filename)[0]
    suffix = suffix.removeprefix("profiles_")
    return suffix


def _resolve_ehr_path(ehr_path: str) -> str:
    """Resolves relative EHR path regardless of current working directory or leading path prefixes."""
    if os.path.isfile(ehr_path):
        return ehr_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    candidates = [
        os.path.join(script_dir, ehr_path),
        os.path.join(project_root, ehr_path),
    ]

    if ehr_path.startswith(("data/", "data\\")):
        stripped = ehr_path[5:]
        candidates.extend(
            [
                os.path.join(script_dir, stripped),
                os.path.join(project_root, stripped),
                stripped,
            ]
        )

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    return ehr_path


def load_profiles(
    ehr_path: str, profiles_path: str | None = None, output_dir: str = "./results"
) -> list[str]:
    """Loads patient profiles from candidate locations based on the EHR file path or an explicit profiles path."""
    ehr_path = _resolve_ehr_path(ehr_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    suffix = _get_ehr_suffix(ehr_path)

    candidates = []
    if profiles_path:
        candidates.append(profiles_path)

    candidates.extend(
        [
            os.path.join(output_dir, f"profiles_{suffix}.json"),
            os.path.join(script_dir, "results", f"profiles_{suffix}.json"),
            os.path.join(project_root, "data", "results", f"profiles_{suffix}.json"),
            os.path.join(os.path.dirname(ehr_path), f"profiles_{suffix}.json"),
            f"profiles_{suffix}.json",
            ehr_path,
        ]
    )

    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "profiles" in data:
                        print(f"Loaded {len(data['profiles'])} profiles from '{path}'")
                        return data["profiles"]
                    elif isinstance(data, list):
                        print(f"Loaded {len(data)} profiles from '{path}'")
                        return data
            except Exception as e:
                print(f"Warning: Failed to load profiles from '{path}': {e}")
                continue

    raise FileNotFoundError(
        f"Could not find profiles file for EHR '{ehr_path}'. "
        f"Checked candidates: {candidates}. "
        "Please generate profiles first using _generate_profile.py."
    )


def load_questions(
    ehr_path: str, questions_path: str | None = None, output_dir: str = "./results"
) -> list[str]:
    """Loads initial patient questions from candidate locations ending with '_questions.json' based on the EHR file path or an explicit questions path."""
    ehr_path = _resolve_ehr_path(ehr_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    suffix = _get_ehr_suffix(ehr_path)
    base_no_ext = os.path.splitext(ehr_path)[0]

    candidates = []
    if questions_path:
        candidates.append(questions_path)

    candidates.extend(
        [
            f"{base_no_ext}_questions.json",
            os.path.join(os.path.dirname(ehr_path), f"{suffix}_questions.json"),
            os.path.join(script_dir, "raw", f"{suffix}_questions.json"),
            os.path.join(project_root, "data", "raw", f"{suffix}_questions.json"),
            os.path.join(output_dir, f"{suffix}_questions.json"),
            os.path.join(script_dir, "results", f"{suffix}_questions.json"),
            os.path.join(project_root, "data", "results", f"{suffix}_questions.json"),
            f"{suffix}_questions.json",
            os.path.join(os.path.dirname(ehr_path), f"questions_{suffix}.json"),
            os.path.join(output_dir, f"questions_{suffix}.json"),
            f"questions_{suffix}.json",
        ]
    )

    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        f.seek(0)
                        content = f.read()
                        import re

                        content_clean = re.sub(r",\s*([\]}])", r"\1", content)
                        data = json.loads(content_clean)

                    if isinstance(data, dict) and "questions" in data:
                        print(
                            f"Loaded {len(data['questions'])} questions from '{path}'"
                        )
                        return data["questions"]
                    elif isinstance(data, list):
                        print(f"Loaded {len(data)} questions from '{path}'")
                        return data
            except Exception as e:
                print(f"Warning: Failed to load questions from '{path}': {e}")
                continue

    if questions_path:
        raise FileNotFoundError(f"Could not find questions file at '{questions_path}'.")

    print(
        f"No questions file found for EHR '{ehr_path}'. Will generate initial question dynamically."
    )
    return []


def _clean_replica(raw_text: str) -> str:
    """Cleans speaker prefixes, markdown artifacts, and quotation marks from patient replicas."""
    cleaned = (raw_text or "").strip()
    cleaned = re.sub(
        r"^(?:\*\*|\*|#)*\s*patient\s*(?:\*\*|\*|#)*\s*:\s*(?:\*\*|\*)*\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip()
    if cleaned.startswith(":"):
        cleaned = cleaned.lstrip(":* \t")
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _is_replica_too_long(replica: str, threshold: int = _REPLICA_THRESHOLD) -> bool:
    """Checks whether a synthetic patient replica exceeds the specified length threshold."""
    norm = replica.lower().strip().strip(".\"'!?:; \t\n\r")
    if (
        norm == "fertig"
        or norm.endswith("\nfertig")
        or norm.endswith(" fertig")
        or norm.startswith("fertig ")
    ):
        return False
    return len(replica) > threshold


def _build_regeneration_prompt(
    history_str: str,
    previous_replica: str,
    threshold: int,
    language: str = "de",
) -> str:
    """Builds a prompt instructing the LLM to re-generate a shorter synthetic patient replica."""
    if language == "en":
        return (
            f"Conversation history:\n{history_str}\n\n"
            f"Your previous response as the patient was too long ({len(previous_replica)} characters, limit is {threshold} characters):\n"
            f'"{previous_replica}"\n\n'
            f"Please re-generate your response as the patient to be much shorter (1–3 short sentences in chat style, under {threshold} characters).\n"
            f"Rules:\n"
            f"- Keep it concise, natural, and in character.\n"
            f"- Do NOT summarize or repeat the chatbot's words.\n"
            f"- If finished, reply only with 'fertig'.\n\n"
            f"Your shortened patient response:"
        )
    elif language == "ru":
        return (
            f"История диалога:\n{history_str}\n\n"
            f"Твой предыдущий ответ как пациента был слишком длинным ({len(previous_replica)} символов, лимит {threshold} символов):\n"
            f'"{previous_replica}"\n\n'
            f"Пожалуйста, перегенерируй свой ответ как пациент значительно короче (1–3 коротких предложения в стиле чата, до {threshold} символов).\n"
            f"Правила:\n"
            f"- Коротко, естественно и в соответствии с ролью пациента.\n"
            f"- НЕ повторяй и не пересказывай слова чат-бота.\n"
            f"- Если диалог окончен, ответь только 'fertig'.\n\n"
            f"Твой сокращенный ответ как пациента:"
        )
    else:  # default 'de'
        return (
            f"Bisheriger Gesprächsverlauf:\n{history_str}\n\n"
            f"Deine vorherige Antwort als Patient war zu lang ({len(previous_replica)} Zeichen, das Limit liegt bei {threshold} Zeichen):\n"
            f'"{previous_replica}"\n\n'
            f"Bitte formuliere deine Antwort als Patient neu, sodass sie deutlich kürzer ist (1–3 kurze Sätze im Chat-Stil, maximal {threshold} Zeichen).\n"
            f"Regeln:\n"
            f"- Sehr kurz, alltagssprachlich und authentisch aus Patientensicht.\n"
            f"- Wiederhole oder fasse keinesfalls zusammen, was der Chatbot gesagt hat.\n"
            f"- Wenn für dich alles geklärt ist, antworte nur mit 'fertig'.\n\n"
            f"Deine neu formulierte, kürzere Antwort als Patient:"
        )


async def _regenerate_replica(
    client: genai.Client,
    previous_replica: str,
    history_str: str,
    system_instruction: str,
    threshold: int = _REPLICA_THRESHOLD,
    language: str = "de",
    model: str = _DEFAULT_MODEL,
    max_output_tokens: int = _MAX_LENGTH,
    max_retries: int = 3,
) -> str:
    """Asks the LLM to re-generate a synthetic patient replica that exceeded the length threshold."""
    current_replica = previous_replica
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7,
        max_output_tokens=max_output_tokens,
    )

    for attempt in range(max_retries):
        if not _is_replica_too_long(current_replica, threshold):
            break

        print(
            f"Patient replica exceeded threshold ({len(current_replica)} > {threshold} chars, "
            f"retry {attempt + 1}/{max_retries}). Asking LLM to re-generate synthetic patient replica..."
        )

        prompt_text = _build_regeneration_prompt(
            history_str=history_str,
            previous_replica=current_replica,
            threshold=threshold,
            language=language,
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt_text,
            config=config,
        )
        cand = _clean_replica(response.text or "")
        if cand:
            current_replica = cand
            if not _is_replica_too_long(current_replica, threshold):
                print(
                    f"Re-generated patient replica within threshold ({len(current_replica)} chars)."
                )
                break

    return current_replica


async def _generate_response(
    profile: str,
    ehr: str,
    client: genai.Client,
    history: str | None = None,
    language: str = "de",
    model: str = _DEFAULT_MODEL,
    max_length: int = _MAX_LENGTH,
    threshold: int = _REPLICA_THRESHOLD,
) -> str:
    system_instruction = _PROMPT_SIMULATE_STEP1.format(profile=profile, ehr=ehr)
    history_str = history if history else "START A NEW CHAT"
    prompt_text = _PROMPT_SIMULATE_STEP2.format(history=history_str)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7,
        max_output_tokens=max_length,
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt_text,
        config=config,
    )
    cleaned = _clean_replica(response.text or "")

    if _is_replica_too_long(cleaned, threshold):
        cleaned = await _regenerate_replica(
            client=client,
            previous_replica=cleaned,
            history_str=history_str,
            system_instruction=system_instruction,
            threshold=threshold,
            language=language,
            model=model,
            max_output_tokens=max_length,
        )

    return cleaned


def _merge_history(messages: list[dict[str, str]]) -> str:
    """Formats chat history for insertion into the prompt for the patient simulator."""
    history = []
    for msg in messages:
        role = msg.get("role", "")
        text = msg.get("text") or msg.get("content", "")
        if role in ("human", "user", "patient"):
            history.append(f"Patient: {text}")
        else:
            history.append(f"Chatbot: {text}")
    return "\n".join(history)


async def login(client: httpx.AsyncClient, username: str, password: str) -> bool:
    """Authenticates with the MAIA backend server and establishes session cookies."""
    try:
        response = await client.post(
            "/api/login", json={"username": username, "password": password}
        )
        if response.status_code == 200:
            print(f"Logged in successfully as '{username}'")
            return True
        else:
            print(f"Login failed (status {response.status_code}): {response.text}")
            return False
    except Exception as e:
        print(f"Error authenticating with server: {e}")
        return False


async def _generate(
    client: httpx.AsyncClient,
    ehr_hint: str,
    profiles: list[str],
    genai_client: genai.Client,
    questions: list[str] | None = None,
    language: str = "de",
    model: str = _DEFAULT_MODEL,
    max_length: int = _MAX_LENGTH,
    threshold: int = _REPLICA_THRESHOLD,
    max_turns: int = _MAX_TURNS,
) -> dict[str, Any]:
    """Executes a single simulation dialogue loop between patient model and backend chatbot."""
    profile_id = random.randint(0, len(profiles) - 1)
    profile = profiles[profile_id]
    chat_history: list[dict[str, str]] = []
    user_id = uuid.uuid4()
    is_first_interaction = True

    initial_question = None
    if questions:
        initial_question = random.choice(questions)

    while True:
        if is_first_interaction and initial_question:
            new_message = initial_question
        else:
            new_message = await _generate_response(
                profile=profile,
                ehr=ehr_hint,
                history=_merge_history(chat_history),
                client=genai_client,
                language=language,
                model=model,
                max_length=max_length,
                threshold=threshold,
            )

        norm_msg = new_message.lower().strip().strip(".\"'!?:; \t\n\r")
        if (
            norm_msg == "fertig"
            or norm_msg.endswith("\nfertig")
            or norm_msg.endswith(" fertig")
            or norm_msg.startswith("fertig ")
        ):
            break

        api_history = (
            [
                {
                    "role": "user"
                    if msg["role"] in ("user", "human", "patient")
                    else "assistant",
                    "content": msg.get("text") or msg.get("content", ""),
                }
                for msg in chat_history
            ]
            if chat_history
            else None
        )
        chat_history.append({"role": "user", "text": new_message})

        chat_payload = {
            "message": new_message,
            "history": api_history,
            "is_first_interaction": is_first_interaction,
            "language": language,
        }

        bot_replied = False
        final_answer = ""

        try:
            async with client.stream(
                "POST", "/api/chat", json=chat_payload
            ) as response:
                if response.status_code == 200:
                    answer_chunks = []
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk_data = json.loads(line)
                            if chunk_data.get("type") == "content":
                                answer_chunks.append(chunk_data.get("content", ""))
                            elif chunk_data.get("type") == "error":
                                print(f"Server error: {chunk_data.get('content')}")
                        except json.JSONDecodeError:
                            continue
                    final_answer = "".join(answer_chunks).strip()
                    if final_answer:
                        print(f"Chatbot: {final_answer}")
                        chat_history.append({"role": "chat", "text": final_answer})
                        bot_replied = True
                else:
                    err_body = await response.aread()
                    print(
                        f"Chat request error ({response.status_code}): {err_body.decode('utf-8', errors='replace')}"
                    )
        except Exception as e:
            print(f"Failed to communicate with backend server: {e}")

        if not bot_replied:
            chat_history.append({"role": "chat", "text": "[no response]"})

        is_first_interaction = False

        turn_limit = max_turns if max_length >= 100 else max_length
        if len(chat_history) >= turn_limit:
            break

    return {
        "chat_history": chat_history,
        "user_id": str(user_id),
        "profile_id": profile_id,
    }


def _load_existing_simulations(sim_json_path: str) -> list[dict[str, Any]]:
    """Loads existing simulations from JSON file if it exists."""
    if not os.path.isfile(sim_json_path):
        return []
    try:
        with open(sim_json_path, "r", encoding="utf-8") as f:
            return json.load(f).get("simulations", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_simulation_results(
    sim_json_path: str,
    sim_csv_path: str,
    simulations: list[dict[str, Any]],
    simulation: dict[str, Any],
) -> None:
    """Saves updated simulations list to JSON and appends the simulation history to CSV."""
    with open(sim_json_path, "w", encoding="utf-8") as f:
        json.dump({"simulations": simulations}, f, ensure_ascii=False, indent=2)
    print(f"Simulation saved to {sim_json_path}")

    user_id = simulation["user_id"]
    profile_id = simulation["profile_id"]
    chat_history = simulation["chat_history"]

    file_exists = os.path.isfile(sim_csv_path)
    with open(sim_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["simulation_id", "profile_id", "speaker", "message"])

        for message in chat_history:
            writer.writerow([user_id, profile_id, message["role"], message["text"]])
    print(f"Simulation history appended to {sim_csv_path}")


async def run(
    client: httpx.AsyncClient,
    ehr_hint: str,
    profiles: list[str],
    genai_client: genai.Client,
    suffix: str,
    questions: list[str] | None = None,
    output_dir: str = "./results",
    language: str = "de",
    model: str = _DEFAULT_MODEL,
    max_length: int = _MAX_LENGTH,
    threshold: int = _REPLICA_THRESHOLD,
) -> dict[str, Any]:
    """Runs a single chat simulation and records the output to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    sim_json_path = os.path.join(output_dir, f"simulations_{suffix}.json")
    sim_csv_path = os.path.join(output_dir, f"simulations_{suffix}.csv")

    simulations = await asyncio.to_thread(_load_existing_simulations, sim_json_path)

    simulation = await _generate(
        client=client,
        ehr_hint=ehr_hint,
        profiles=profiles,
        genai_client=genai_client,
        questions=questions,
        language=language,
        model=model,
        max_length=max_length,
        threshold=threshold,
    )
    simulations.append(simulation)

    await asyncio.to_thread(
        _save_simulation_results,
        sim_json_path,
        sim_csv_path,
        simulations,
        simulation,
    )

    return simulation


async def main(
    ehr_path: str,
    server_url: str = "http://localhost:8080",
    username: str | None = None,
    password: str | None = None,
    language: str = "de",
    num_simulations: int = 20,
    model: str = _DEFAULT_MODEL,
    project: str = "kuligin-sandbox-502813",
    profiles_path: str | None = None,
    questions_path: str | None = None,
    output_dir: str = "./results",
    max_length: int = _MAX_LENGTH,
    threshold: int = _REPLICA_THRESHOLD,
):
    """Main orchestration function to run patient chat simulations in batch."""
    username = (
        username
        or os.getenv("MAIA_USERNAME")
        or os.getenv("MAIA_LOGIN")
        or os.getenv("APP_USERNAME")
        or "admin_maia_secure"
    )
    password = (
        password
        or os.getenv("MAIA_PASSWORD")
        or os.getenv("APP_PASSWORD")
        or "super_secret_password_2026"
    )
    ehr_path = _resolve_ehr_path(ehr_path)
    profiles = load_profiles(
        ehr_path, profiles_path=profiles_path, output_dir=output_dir
    )
    questions = load_questions(
        ehr_path, questions_path=questions_path, output_dir=output_dir
    )
    ehr_hint = create_ehr_hint(ehr_path)
    suffix = _get_ehr_suffix(ehr_path)

    genai_client = genai.Client(
        http_options=types.HttpOptions(api_version="v1"),
        vertexai=True,
        project=project,
        location="global",
    )

    async with httpx.AsyncClient(base_url=server_url, timeout=120.0) as client:
        logged_in = await login(client, username, password)
        if not logged_in:
            print("Aborting simulation run due to login failure.")
            return

        for i in range(num_simulations):
            print(f"\n--- Starting simulation {i + 1}/{num_simulations} ---")
            await run(
                client=client,
                ehr_hint=ehr_hint,
                profiles=profiles,
                genai_client=genai_client,
                suffix=suffix,
                questions=questions,
                output_dir=output_dir,
                language=language,
                model=model,
                max_length=max_length,
                threshold=threshold,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate patient chat sessions with MAIA backend API"
    )
    parser.add_argument("--ehr_path", help="Path to the EHR JSON file", required=True)
    parser.add_argument(
        "--profiles_path", default=None, help="Optional path to profiles JSON file"
    )
    parser.add_argument(
        "--questions_path", default=None, help="Optional path to questions JSON file"
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory for output simulations and CSVs",
    )
    parser.add_argument(
        "--server_url",
        default=os.getenv("MAIA_SERVER_URL", "http://localhost:8080"),
        help="MAIA backend base URL",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("MAIA_USERNAME")
        or os.getenv("MAIA_LOGIN")
        or os.getenv("APP_USERNAME")
        or "admin_maia_secure",
        help="Backend username (reads MAIA_USERNAME, MAIA_LOGIN, or APP_USERNAME environment variables)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("MAIA_PASSWORD")
        or os.getenv("APP_PASSWORD")
        or "super_secret_password_2026",
        help="Backend password (reads MAIA_PASSWORD or APP_PASSWORD environment variables)",
    )
    parser.add_argument("--language", default="de", help="Language code (de, en, ru)")
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=20,
        help="Number of simulations to execute",
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument(
        "--project",
        default=os.getenv("GCP_PROJECT", "kuligin-sandbox-502813"),
        help="GCP project ID",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=_MAX_LENGTH,
        help="Maximum generation token length for patient simulator",
    )
    parser.add_argument(
        "--threshold",
        "--replica_threshold",
        dest="threshold",
        type=int,
        default=_REPLICA_THRESHOLD,
        help="Length threshold in characters above which a patient replica is re-generated by LLM",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            ehr_path=args.ehr_path,
            profiles_path=args.profiles_path,
            questions_path=args.questions_path,
            output_dir=args.output_dir,
            server_url=args.server_url,
            username=args.username,
            password=args.password,
            language=args.language,
            num_simulations=args.num_simulations,
            model=args.model,
            project=args.project,
            max_length=args.max_length,
            threshold=args.threshold,
        )
    )
