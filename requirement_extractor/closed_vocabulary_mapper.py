from __future__ import annotations

import re
import unicodedata
from typing import List, Optional


def _normalize(text: str) -> str:
    """
    Produit une forme stable pour la détection lexicale.

    Cette normalisation :
    - met le texte en minuscules ;
    - supprime les accents sans modifier le sens ;
    - uniformise les apostrophes et les espaces ;
    - conserve les signes utiles aux négations.
    """

    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    normalized = normalized.lower()
    normalized = normalized.replace("’", "'").replace("`", "'")
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def _contains(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


# =====================================================================
# ACCESS TYPE
# =====================================================================


_ACCESS_PATTERNS = {
    "sequential": (
        r"\bsequentiel(?:le|s)?\b",
        r"\bsequential(?:ly)?\b",
        r"\bseq(?:uential)?\s*(?:io|i/o|access)?\b",
    ),
    "random": (
        r"\baleatoire(?:s)?\b",
        r"\brandom(?:ly)?\b",
        r"\brndm\b",
    ),
    "parallel": (
        r"\bparallele(?:s)?\b",
        r"\bparallel(?:ism|ly)?\b",
        r"\bparallel\s*(?:io|i/o|access)?\b",
    ),
    "streaming": (
        r"\bstreaming\b",
        r"\bstream(?:ed|ing)?\s*(?:io|i/o|access)?\b",
    ),
    "mixed": (
        r"\bmixte(?:s)?\b",
        r"\bmixed\b",
        r"\bhybride(?:s)?\b",
        r"\bhybrid\b",
        r"\bmixd\b",
    ),
}


def detect_access_types(text: str) -> List[str]:
    """
    Retourne tous les types d'accès explicitement détectés.

    Plusieurs valeurs différentes sont conservées. Cette propriété permet au
    StateGuard de détecter un conflit au lieu de choisir silencieusement une
    valeur.

    Une formulation explicitement mixte est représentée uniquement par
    ``mixed``.
    """

    normalized = _normalize(text)

    explicit_mixed = any(
        _contains(pattern, normalized)
        for pattern in _ACCESS_PATTERNS["mixed"]
    )

    if explicit_mixed:
        return ["mixed"]

    detected: List[str] = []

    for access_type in (
        "sequential",
        "random",
        "parallel",
        "streaming",
    ):
        if any(
            _contains(pattern, normalized)
            for pattern in _ACCESS_PATTERNS[access_type]
        ):
            detected.append(access_type)

    return detected


def map_access_type(text: str) -> Optional[str]:
    """
    Mappe une formulation non ambiguë vers le vocabulaire fermé :

    ``sequential``, ``random``, ``parallel``, ``streaming`` ou ``mixed``.

    En présence de plusieurs types distincts sans marqueur explicite de
    mélange, la fonction retourne ``None``. Le nouvel extracteur peut utiliser
    ``detect_access_types`` pour conserver toutes les valeurs et laisser le
    StateGuard déclarer le conflit.
    """

    detected = detect_access_types(text)

    if len(detected) == 1:
        return detected[0]

    return None


# =====================================================================
# HIGH AVAILABILITY
# =====================================================================


_HA_FALSE_PATTERNS = (
    # Négation placée avant HA.
    r"\b(?:sans|without|no)\s+(?:la\s+)?(?:ha|high availability|"
    r"haute disponibilite|haute dispo)\b",
    r"\bpas\s+de\s+(?:ha|haute disponibilite|haute dispo)\b",

    # Négation ou désactivation placée après HA.
    r"\b(?:ha|high availability|haute disponibilite|haute dispo)\s*"
    r"(?:[:=]\s*)?(?:non|no|false|off|desactivee?|disabled)\b",

    # Caractère obligatoire/requis explicitement nié.
    r"\b(?:ha|high availability|haute disponibilite|haute dispo)\b"
    r".{0,25}\b(?:non|pas|not)\s+"
    r"(?:requise?|required|obligatoire|necessaire|needed)\b",

    # Formulation inverse : non requise pour HA.
    r"\b(?:non|pas|not)\s+"
    r"(?:requise?|required|obligatoire|necessaire|needed)\b"
    r".{0,25}\b(?:ha|high availability|haute disponibilite|"
    r"haute dispo)\b",

    # HA explicitement facultative.
    r"\b(?:ha|high availability|haute disponibilite|haute dispo)\b"
    r".{0,20}\b(?:optionnelle?|optional|facultative)\b",
)


_HA_TRUE_BOOLEAN_PATTERNS = (
    r"\b(?:ha|high availability|haute disponibilite|haute dispo)\s*"
    r"(?:[:=]\s*)?(?:oui|yes|true|on)\b",
)


_HA_POSITIVE_REQUIREMENT_PATTERNS = (
    r"\b(?:ha|high availability|haute disponibilite|haute dispo)\b"
    r".{0,25}\b(?:requise?|required|obligatoire|necessaire|needed)\b",

    r"\b(?:requise?|required|obligatoire|necessaire|needed)\b"
    r".{0,25}\b(?:ha|high availability|haute disponibilite|"
    r"haute dispo)\b",
)


_HA_POSITIVE_CONCEPT_PATTERNS = (
    r"\b(?:tolerance aux pannes|fault tolerance)\b",
    r"\b(?:redondance|redundancy)\b",
)


_HA_BROAD_POSITIVE_PATTERNS = (
    r"\b(?:haute disponibilite|haute dispo|high availability)\b",
)


def _has_non_negated_match(
    patterns: tuple[str, ...],
    text: str,
    context_size: int = 18,
) -> bool:
    """
    Recherche une formulation positive sans laisser une négation appartenant
    à une proposition suivante annuler cette affirmation.

    La vérification porte sur :
    - le texte réellement capturé par le motif ;
    - un petit contexte placé uniquement avant ce motif.

    Exemple :
    ``HA obligatoire mais sans HA`` contient deux propositions indépendantes.
    La première doit produire ``True`` et la seconde ``False``.

    À l'inverse :
    - ``HA non requise`` ;
    - ``sans HA obligatoire`` ;
    - ``pas de redondance``

    restent correctement reconnues comme négatives.
    """

    negation_pattern = (
        r"\b(?:non|no|pas|not|sans|without|false|off|"
        r"desactivee?|disabled)\b"
    )

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            prefix_start = max(0, match.start() - context_size)
            prefix = text[prefix_start:match.start()]
            matched_text = match.group(0)

            local_context = f"{prefix} {matched_text}".strip()

            if re.search(
                negation_pattern,
                local_context,
                flags=re.IGNORECASE,
            ):
                continue

            return True

    return False


def detect_ha_values(text: str) -> List[bool]:
    """
    Détecte toutes les valeurs HA explicitement présentes.

    Retour :
    - ``[True]`` : HA demandée ;
    - ``[False]`` : HA non demandée ;
    - ``[False, True]`` : contradiction explicite ;
    - ``[]`` : aucune décision HA suffisamment claire.
    """

    normalized = _normalize(text)

    has_false = any(
        _contains(pattern, normalized)
        for pattern in _HA_FALSE_PATTERNS
    )

    explicit_true = any(
        _contains(pattern, normalized)
        for pattern in _HA_TRUE_BOOLEAN_PATTERNS
    )

    positive_requirement = _has_non_negated_match(
        _HA_POSITIVE_REQUIREMENT_PATTERNS,
        normalized,
    )

    positive_concept = _has_non_negated_match(
        _HA_POSITIVE_CONCEPT_PATTERNS,
        normalized,
    )

    broad_positive = any(
        _contains(pattern, normalized)
        for pattern in _HA_BROAD_POSITIVE_PATTERNS
    )

    has_true = (
        explicit_true
        or positive_requirement
        or positive_concept
        or (broad_positive and not has_false)
    )

    # Analyse locale de la proposition HA pour préserver les contradictions
    # explicites comme « HA oui et non » ou « HA yes and no ».
    ha_match = re.search(
        r"\b(?:ha|high availability|haute disponibilite|haute dispo)\b"
        r"(?P<tail>.{0,45})",
        normalized,
    )

    if ha_match:
        tail = ha_match.group("tail")

        if re.search(
            r"\b(?:oui|yes|true|on)\b",
            tail,
        ):
            has_true = True

        if re.search(
            r"\b(?:non|no|false|off|disabled|desactivee?)\b",
            tail,
        ):
            has_false = True

    standalone_ha = _contains(r"\bha\b", normalized)

    values: List[bool] = []

    if has_false:
        values.append(False)

    if has_true:
        values.append(True)

    if not values and standalone_ha:
        values.append(True)

    return values


def map_ha_required(text: str) -> Optional[bool]:
    """
    Retourne la valeur HA uniquement lorsqu'elle est non ambiguë.

    Les contradictions sont volontairement laissées non résolues. Le nouvel
    extracteur utilise ``detect_ha_values`` afin de créer deux candidats
    distincts et permettre au StateGuard de signaler le conflit.
    """

    detected = detect_ha_values(text)

    if len(detected) == 1:
        return detected[0]

    return None
