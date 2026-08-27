from __future__ import annotations
import re, unicodedata

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", text).strip()

def fold(text: str) -> str:
    normalized = normalize_text(text).lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", normalized)
        if unicodedata.category(c) != "Mn"
    )

def is_short_affirmation(text: str) -> bool:
    return fold(text).strip(" .!?") in {
        "yes", "y", "oui", "ouais", "yeah", "yep",
        "required", "mandatory", "obligatoire",
    }

def is_short_negation(text: str) -> bool:
    return fold(text).strip(" .!?") in {
        "no", "n", "non", "nope",
        "not required", "pas obligatoire", "pas necessaire",
    }
