from __future__ import annotations

import re
import unicodedata


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)

    return "".join(
        char
        for char in normalized
        if unicodedata.category(char) != "Mn"
    )


def normalize_text(text: str) -> str:
    """
    Prépare le texte utilisateur avant l'extraction.

    Objectifs :
    - uniformiser les espaces ;
    - gérer les décimales françaises ;
    - séparer les unités collées aux nombres ;
    - séparer les mots collés aux nombres ;
    - normaliser quelques écritures fréquentes ;
    - garder le sens original du message.
    """

    if not text:
        return ""

    text = text.strip()
    text = text.replace("\u00a0", " ")

    # Espaces multiples
    text = re.sub(r"\s+", " ", text)

    # Décimales françaises :
    # 750,5 TiB -> 750.5 TiB
    # 1,5 GB -> 1.5 GB
    text = re.sub(
        r"(\d),(\d)",
        r"\1.\2",
        text,
    )

    # Mot collé à un nombre :
    # idéal50000 -> idéal 50000
    # ideal50000 -> ideal 50000
    text = re.sub(
        r"([A-Za-zÀ-ÿ])(\d)",
        r"\1 \2",
        text,
    )

    # Nombre collé à une unité ou un mot :
    # 75000USD -> 75000 USD
    # 9kW -> 9 kW
    # 80GB/s -> 80 GB/s
    text = re.sub(
        r"(\d)(tib|tb|gib|gb/s|gbps|gbs|gb|mib|mb|kw|w|usd|dollars?|\$|%)\b",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    replacements = {
        "gbps": "GB/s",
        "gbs": "GB/s",
        "go/s": "GB/s",
        "giga par seconde": "GB/s",

        "ecriture": "écriture",
        "sequentiel": "séquentiel",
        "aleatoire": "aléatoire",
        "parallele": "parallèle",
        "disponibilite": "disponibilité",

        "noeud": "noeud",
        "noeuds": "noeuds",
    }

    for old, new in replacements.items():
        text = re.sub(
            re.escape(old),
            new,
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalized_for_matching(text: str) -> str:
    """
    Version minuscule sans accents pour faciliter les regex.
    """

    return strip_accents(normalize_text(text)).lower()