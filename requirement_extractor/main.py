from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Important :
# Le .env doit être chargé avant la création du chatbot.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPOSITORY_ROOT / ".env")

from .models import ChatbotStatus
from .requirement_chatbot import RequirementChatbot


def llm_status_text() -> str:
    enabled = os.getenv("ENABLE_LLM_FALLBACK", "false").lower() == "true"
    model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    if enabled:
        return f"enabled | model={model} | host={host}"

    return "disabled"


def final_json_as_plain_dict(bot: RequirementChatbot) -> dict:
    state = bot.state

    return {
        key: None if value is None else value.value
        for key, value in state.final_json.items()
    }


def print_state(bot: RequirementChatbot) -> None:
    state = bot.state

    print("\n" + "=" * 80)
    print(f"Stage  : {state.stage.value}")
    print(f"Status : {state.status.value}")

    print("\nJSON courant :")

    plain_json = final_json_as_plain_dict(bot)

    print(
        json.dumps(
            plain_json,
            indent=2,
            ensure_ascii=False,
        )
    )

    if state.status == ChatbotStatus.NEEDS_CLARIFICATION:
        print("\nQuestions :")

        if state.questions:
            for index, question in enumerate(state.questions, start=1):
                print(f"{index}. {question}")
        else:
            print("Aucune question générée.")

    if state.calculation_result:
        print("\nCalcul :")

        print(
            json.dumps(
                state.calculation_result,
                indent=2,
                ensure_ascii=False,
            )
        )

    print("=" * 80 + "\n")


def print_startup_banner() -> None:
    print("Hybrid Lustre Requirement Chatbot v2")
    print("=" * 80)
    print(f"LLM fallback : {llm_status_text()}")
    print("\nCommandes :")
    print("- exit  : quitter")
    print("- reset : commencer un nouveau besoin")
    print("=" * 80 + "\n")


def main() -> None:
    bot = RequirementChatbot()

    print_startup_banner()

    while True:
        try:
            user_text = input("User> ").strip()
        except KeyboardInterrupt:
            print("\nArrêt demandé.")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("Fin de session.")
            break

        if user_text.lower() in {"reset", "restart", "new"}:
            bot.reset()
            print("\n[RESET] Nouvelle conversation commencée.\n")
            continue

        bot.process_user_message(user_text)
        print_state(bot)


if __name__ == "__main__":
    main()