from __future__ import annotations

import json

from .requirement_chatbot import RequirementChatbot


tests = [
    (
        "On veut 500 TiB utiles, 200 clients, fichiers moyens 2 GB, "
        "max 100 GB, 10 millions de fichiers, ratio 70/30, accès mixte, "
        "lecture 80 GB/s, écriture 40 GB/s, HA obligatoire, "
        "budget 100000 USD, puissance 15 kW, croissance 30%."
    ),
    (
        "Le cluster a été acheté en 2024 et nous avons 3 salles machines. "
        "Je veux une solution Lustre robuste."
    ),
    (
        "Besoin 100 TiB utiles, 40 clients, fichiers moyens 2 GB, "
        "max 20 GB, 1 million fichiers, ratio 80% lecture et 50% écriture, "
        "accès random, lecture 10 GB/s, écriture 10 GB/s, HA oui, "
        "budget 50000 USD, puissance 5000 W, croissance 15%."
    ),
]


def main():
    for index, text in enumerate(tests, start=1):
        print("\n" + "#" * 80)
        print(f"TEST {index}")
        print(text)

        bot = RequirementChatbot()
        bot.process_user_message(text)

        print(
            json.dumps(
                bot.state.to_dict(),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()