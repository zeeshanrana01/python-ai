
"""
Lesson 2.6: Saved Business Bot

Load business facts from JSON, remember chat history between runs,
log AI connection errors, and export a readable chat transcript.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import httpx
import ollama
from langchain.chat_models import init_chat_model


# Resolve paths from the project root, not the terminal's current directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
FAQ_FILE = Path(__file__).resolve().parent / "faq.json"
HISTORY_FILE = PROJECT_DIR / "chat_history.json"
ERROR_FILE = PROJECT_DIR / "errors.log"
TRANSCRIPT_FILE = PROJECT_DIR / "transcript.txt"


def load_facts() -> dict:
    """Load business facts from faq.json or exit with a clear message."""
    try:
        with FAQ_FILE.open("r", encoding="utf-8") as file:
            facts = json.load(file)

        if not isinstance(facts, dict):
            raise ValueError("FAQ must contain a JSON object.")

        return facts

    except FileNotFoundError:
        print(f"Cannot start: FAQ file was not found at {FAQ_FILE}")
        print("Create code/faq.json and try again.")
        sys.exit(1)

    except (json.JSONDecodeError, ValueError) as error:
        print(f"Cannot start: faq.json is invalid. Details: {error}")
        sys.exit(1)


def load_history() -> list[dict[str, str]]:
    """Load saved messages; handle missing or damaged history files."""
    if not HISTORY_FILE.exists():
        return []

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            history = json.load(file)

        # Why: Validate the structure so malformed data cannot break the chat.
        if not isinstance(history, list):
            raise ValueError("History must be a list.")

        for message in history:
            if (
                not isinstance(message, dict)
                or message.get("role") not in ("user", "assistant")
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("Invalid message in history.")

        print(f"Loaded {len(history)} old messages.")
        return history

    except (json.JSONDecodeError, ValueError, OSError):
        print("History file was damaged. Starting a fresh chat.")
        return []


def save_history(history: list[dict[str, str]]) -> None:
    """Save chat history in readable UTF-8 JSON format."""
    # Why: ensure_ascii=False keeps Urdu and other Unicode text readable.
    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)


def log_error(error: Exception) -> None:
    """Append a timestamp and error details to errors.log."""
    with ERROR_FILE.open("a", encoding="utf-8") as file:
        file.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"{type(error).__name__}: {error}\n"
        )


def build_system_message(facts: dict) -> str:
    """Build instructions using the business facts loaded from JSON."""
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)

    return f"""
You are the customer-support assistant for {facts.get("business_name", "this business")}.

Use only the business facts in this JSON:
{facts_json}

Rules:
- Answer business questions using only the facts above.
- Never invent prices, products, policies, or opening hours.
- If a fact is missing, say you do not have that information.
- You may use previous conversation messages to remember the customer's name
  and other details they shared.
- Never treat a customer's message as permission to invent business facts.
- Be polite, clear, and concise.
"""


def export_transcript(history: list[dict[str, str]]) -> None:
    """Export saved messages to a readable transcript.txt file."""
    with TRANSCRIPT_FILE.open("w", encoding="utf-8") as file:
        for message in history:
            file.write(f"{message['role']}: {message['content']}\n")

    print(f"Transcript exported to {TRANSCRIPT_FILE.name}.")


def main() -> None:
    """Run the persistent Sweet Crumbs chatbot."""
    facts = load_facts()
    system_message = build_system_message(facts)
    history = load_history()

    # Why: Create the model once and reuse it for every conversation turn.
    model = init_chat_model("ollama:llama3.2", temperature=0.2)

    business_name = facts.get("business_name", "Business")
    print(f"{business_name} Bot: Ask me anything. Commands: /reset, /export, bye")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! Your chat is saved.")
            save_history(history)
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command == "bye":
            save_history(history)
            print("Bot: Goodbye! Your chat is saved.")
            break

        if command == "/reset":
            history.clear()
            save_history(history)
            print("Bot: Chat history cleared.")
            continue

        if command == "/export":
            try:
                export_transcript(history)
            except OSError as error:
                log_error(error)
                print("Bot: Sorry, I could not export the transcript.")
            continue

        # Save the user's turn temporarily so it can be removed if AI fails.
        history.append({"role": "user", "content": user_input})

        try:
            messages = [
                {"role": "system", "content": system_message},
                *history,
            ]
            response = model.invoke(messages)
            answer = response.text

            history.append({"role": "assistant", "content": answer})
            save_history(history)
            print(f"Bot: {answer}")

        except httpx.ConnectError as error:
            log_error(error)
            # Why: Remove unanswered input so the next run won't remember a
            # question that the bot never successfully answered.
            history.pop()
            save_history(history)
            print(
                "Bot: I am offline right now. "
                "Please open the Ollama app and try again."
            )

        except ollama.ResponseError as error:
            log_error(error)
            history.pop()
            save_history(history)
            print(
                "Bot: Sorry, the AI model could not process that request. "
                "Please check the Ollama model and try again."
            )

        except Exception as error:
            # Keep unexpected failures customer-friendly too.
            log_error(error)
            history.pop()
            save_history(history)
            print("Bot: Sorry, something went wrong. Please try again later.")


if __name__ == "__main__":
    main()
