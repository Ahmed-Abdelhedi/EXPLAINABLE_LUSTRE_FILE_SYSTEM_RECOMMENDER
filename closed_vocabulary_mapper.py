from __future__ import annotations

from typing import Optional


def map_access_type(text: str) -> Optional[str]:
    """
    Mappe les expressions utilisateur vers :
    sequential, random, parallel, streaming, mixed.
    """

    text = text.lower()

    if any(term in text for term in ["mixte", "mixed", "hybride"]):
        return "mixed"

    if (
        any(term in text for term in ["parallèle", "parallele", "parallel"])
        and any(term in text for term in ["séquentiel", "sequentiel", "sequential"])
    ):
        return "mixed"

    if any(term in text for term in ["séquentiel", "sequentiel", "sequential"]):
        return "sequential"

    if "gros fichiers" in text:
        return "sequential"

    if any(term in text for term in ["aléatoire", "aleatoire", "random"]):
        return "random"

    if any(term in text for term in ["parallèle", "parallele", "parallel"]):
        return "parallel"

    if "streaming" in text:
        return "streaming"

    return None


def map_ha_required(text: str) -> Optional[bool]:
    """
    Détecte si HA est requise ou non.

    Les négations sont testées avant les expressions positives.
    """

    text = text.lower()

    false_markers = [
        "pas de ha",
        "sans ha",
        "ha non obligatoire",
        "ha pas obligatoire",
        "no ha",
        "without ha",
        "pas obligatoire",
        "not required",
    ]

    true_markers = [
        "ha obligatoire",
        "avec ha",
        "ha yes",
        "high availability required",
        "haute disponibilité",
        "haute disponibilite",
        "oui ha",
        "redondance",
        "redondant",
        "tolérance aux pannes",
        "tolerance aux pannes",
        "fiable",
        "fiabilité",
        "fiabilite",
        "robuste",
        "reliable",
    ]

    if any(marker in text for marker in false_markers):
        return False

    if any(marker in text for marker in true_markers):
        return True

    return None