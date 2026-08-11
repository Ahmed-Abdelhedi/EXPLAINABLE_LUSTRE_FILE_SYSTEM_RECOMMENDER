from __future__ import annotations

import re
import unicodedata


_MINUS_VARIANTS = (
    "\u2212",  # minus sign
    "\u2010",  # hyphen
    "\u2011",  # non-breaking hyphen
    "\u2012",  # figure dash
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\ufe63",  # small hyphen-minus
    "\uff0d",  # fullwidth hyphen-minus
)

_PLUS_VARIANTS = (
    "\uff0b",  # fullwidth plus
)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)

    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )


def _normalize_numeric_token(match: re.Match[str]) -> str:
    """
    Normalise un token numérique sans perdre son signe.

    Exemples :
    - 1,5        -> 1.5
    - -10,5      -> -10.5
    - 10,000,000 -> 10000000
    - 100 000    -> 100000
    - 1_000_000  -> 1000000

    Heuristique pour une virgule unique :
    - partie fractionnaire de 1 ou 2 chiffres : décimale ;
    - trois chiffres après la virgule : séparateur de milliers,
      sauf lorsque la partie entière vaut zéro.
    """

    token = match.group(0)

    sign = ""
    unsigned = token

    if unsigned.startswith(("+", "-")):
        sign = unsigned[0]
        unsigned = unsigned[1:]

    unsigned = unsigned.replace("_", "")

    # Espaces internes employés comme séparateurs de milliers.
    unsigned = re.sub(r"(?<=\d)\s+(?=\d)", "", unsigned)

    comma_count = unsigned.count(",")
    dot_count = unsigned.count(".")

    if comma_count > 1 and dot_count == 0:
        unsigned = unsigned.replace(",", "")

    elif dot_count > 1 and comma_count == 0:
        unsigned = unsigned.replace(".", "")

    elif comma_count and dot_count:
        last_comma = unsigned.rfind(",")
        last_dot = unsigned.rfind(".")

        if last_comma > last_dot:
            unsigned = unsigned.replace(".", "")
            unsigned = unsigned.replace(",", ".")
        else:
            unsigned = unsigned.replace(",", "")

    elif comma_count == 1:
        integer_part, fractional_part = unsigned.split(",", 1)

        if (
            len(fractional_part) == 3
            and integer_part not in {"0", "00", "000"}
        ):
            unsigned = integer_part + fractional_part
        else:
            unsigned = integer_part + "." + fractional_part

    return sign + unsigned


def _normalize_numbers(text: str) -> str:
    """
    Normalise les nombres signés, décimaux et groupés.
    """

    number_pattern = re.compile(
        r"(?<![\w.])"
        r"[+-]?"
        r"(?:"
        r"\d{1,3}(?:[ _.,]\d{3})+"
        r"|"
        r"\d+(?:[.,]\d+)?"
        r")"
        r"(?![\w.])"
    )

    return number_pattern.sub(_normalize_numeric_token, text)


def normalize_text(text: str) -> str:
    """
    Prépare le texte utilisateur avant l'extraction.

    Objectifs :
    - préserver et normaliser les signes + et - ;
    - distinguer décimales et séparateurs de milliers ;
    - uniformiser les espaces ;
    - séparer les unités collées aux nombres ;
    - conserver le sens original du message ;
    - éviter toute correction métier silencieuse.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.strip()

    text = (
        text.replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2007", " ")
    )

    for character in _MINUS_VARIANTS:
        text = text.replace(character, "-")

    for character in _PLUS_VARIANTS:
        text = text.replace(character, "+")

    text = (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("´", "'")
    )

    # "- 500 TiB" -> "-500 TiB"
    # "+ 30 %"    -> "+30 %"
    text = re.sub(
        r"(?<!\w)([+-])\s+(?=\d)",
        r"\1",
        text,
    )

    text = _normalize_numbers(text)

    # ideal50000 -> ideal 50000
    # budget-500 -> budget -500
    text = re.sub(
        r"([A-Za-zÀ-ÿ])(?=[+-]?\d)",
        r"\1 ",
        text,
    )

    # 75000USD -> 75000 USD
    # -9kW     -> -9 kW
    # +30%     -> +30 %
    unit_pattern = (
        r"tib|tb|gib|gb/s|gbps|gbs|gb|"
        r"mib|mb/s|mbps|mb|"
        r"kw|mw|w|"
        r"usd|dollars?|"
        r"%"
    )

    text = re.sub(
        rf"(?<=\d)(?P<unit>{unit_pattern})(?![A-Za-zÀ-ÿ])",
        r" \g<unit>",
        text,
        flags=re.IGNORECASE,
    )

    # 100000$ -> 100000 $
    text = re.sub(
        r"(?<=\d)\$",
        " $",
        text,
    )

    replacements = {
        "gbps": "GB/s",
        "gbs": "GB/s",
        "go/s": "GB/s",
        "giga par seconde": "GB/s",
        "mbps": "MB/s",

        "ecriture": "écriture",
        "sequentiel": "séquentiel",
        "aleatoire": "aléatoire",
        "parallele": "parallèle",
        "disponibilite": "disponibilité",
    }

    for old, new in replacements.items():
        text = re.sub(
            re.escape(old),
            new,
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*:\s*", ":", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalized_for_matching(text: str) -> str:
    """
    Version minuscule sans accents destinée aux expressions régulières.

    Les signes numériques sont préservés.
    """

    return strip_accents(normalize_text(text)).lower()
