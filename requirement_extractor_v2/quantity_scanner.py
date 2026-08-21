from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple

from quantulum3 import parser as quantulum_parser
from symspellpy import SymSpell, Verbosity

from .models import (
    Quantity,
    QuantityDetection,
    QuantityDimension,
)


# =====================================================================
# INTERNAL MATCH REPRESENTATION
# =====================================================================


@dataclass(frozen=True)
class _DetectedQuantity:
    raw: str
    normalized: str
    value: int | float
    unit: Optional[str]
    dimension: QuantityDimension
    start: int
    end: int
    detection: QuantityDetection
    corrected: bool = False


# =====================================================================
# NUMBER PARSING
# =====================================================================


_NUMBER_PATTERN = (
    r"[+-]?"
    r"(?:"
    r"\d{1,3}(?:[ _.,]\d{3})+"
    r"|"
    r"\d+(?:[.,]\d+)?"
    r")"
)

_SCALE_PATTERN = (
    r"(?:"
    r"k|"
    r"thousand|thousands|"
    r"mille|milles|"
    r"m|million|millions|"
    r"b|billion|billions|"
    r"milliard|milliards"
    r")\b"
)


def _normalize_numeric_literal(token: str) -> Decimal:
    """
    Convert a textual digit-based number to Decimal.

    Supported examples:
    - 500
    - -500
    - 12.5
    - 12,5
    - 100 000
    - 100_000
    - 10,000,000
    """

    value = token.strip().replace("_", "")

    sign = ""
    if value.startswith(("+", "-")):
        sign = value[0]
        value = value[1:]

    value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)

    comma_count = value.count(",")
    dot_count = value.count(".")

    if comma_count > 1 and dot_count == 0:
        value = value.replace(",", "")

    elif dot_count > 1 and comma_count == 0:
        value = value.replace(".", "")

    elif comma_count and dot_count:
        last_comma = value.rfind(",")
        last_dot = value.rfind(".")

        if last_comma > last_dot:
            value = value.replace(".", "")
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")

    elif comma_count == 1:
        integer_part, fractional_part = value.split(",", 1)

        if (
            len(fractional_part) == 3
            and integer_part not in {"0", "00", "000"}
        ):
            value = integer_part + fractional_part
        else:
            value = integer_part + "." + fractional_part

    try:
        return Decimal(sign + value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid numeric literal: {token!r}") from exc


def _scale_multiplier(scale: Optional[str]) -> Decimal:
    if not scale:
        return Decimal("1")

    normalized = scale.strip().lower().rstrip(".")

    if normalized in {
        "k",
        "thousand",
        "thousands",
        "mille",
        "milles",
    }:
        return Decimal("1000")

    if normalized in {
        "m",
        "million",
        "millions",
    }:
        return Decimal("1000000")

    if normalized in {
        "b",
        "billion",
        "billions",
        "milliard",
        "milliards",
    }:
        return Decimal("1000000000")

    return Decimal("1")


def _to_python_number(value: Decimal | int | float) -> int | float:
    decimal_value = (
        value
        if isinstance(value, Decimal)
        else Decimal(str(value))
    )

    if decimal_value == decimal_value.to_integral_value():
        return int(decimal_value)

    return float(decimal_value)


def _parse_number(
    number_token: str,
    scale_token: Optional[str] = None,
) -> int | float:
    value = _normalize_numeric_literal(number_token)
    value *= _scale_multiplier(scale_token)
    return _to_python_number(value)


# =====================================================================
# UNIT DEFINITIONS
# =====================================================================


# Longer tokens must appear before shorter tokens.
_UNIT_RULES: Tuple[
    Tuple[re.Pattern[str], str, QuantityDimension],
    ...
] = (
    # Throughput
    # IMPORTANT:
    # Byte/s and bit/s throughput units are case-sensitive concepts.
    # Do NOT compile these four rules with re.IGNORECASE, otherwise
    # "Gbps" can be incorrectly matched by the "GBps" byte/s rule.
    (
        re.compile(
            r"(?:GB/s|GBps|GBPS|gb/s|gbs|Go/s)"
        ),
        "GB/s",
        QuantityDimension.THROUGHPUT,
    ),
    (
        re.compile(
            r"(?:Gbps|gbps|Gbit/s|Gbits/s|gbit/s|gbits/s)"
        ),
        "Gbps",
        QuantityDimension.THROUGHPUT,
    ),
    (
        re.compile(
            r"(?:MB/s|MBps|MBPS|mb/s|mbs|Mo/s)"
        ),
        "MB/s",
        QuantityDimension.THROUGHPUT,
    ),
    (
        re.compile(
            r"(?:Mbps|mbps|Mbit/s|Mbits/s|mbit/s|mbits/s)"
        ),
        "Mbps",
        QuantityDimension.THROUGHPUT,
    ),

    # Capacity
    (
        re.compile(
            r"(?:TiB|tebibytes?|tebioctets?)",
            re.IGNORECASE,
        ),
        "TiB",
        QuantityDimension.CAPACITY,
    ),
    (
        re.compile(
            r"(?:TB|terabytes?|t[ée]raoctets?)",
            re.IGNORECASE,
        ),
        "TB",
        QuantityDimension.CAPACITY,
    ),

    # File/storage sizes
    (
        re.compile(
            r"(?:GiB|gibibytes?)",
            re.IGNORECASE,
        ),
        "GiB",
        QuantityDimension.FILE_SIZE,
    ),
    (
        re.compile(
            r"(?:GB|gigabytes?|giga(?:s)?|Go)",
            re.IGNORECASE,
        ),
        "GB",
        QuantityDimension.FILE_SIZE,
    ),
    (
        re.compile(
            r"(?:MiB|mebibytes?)",
            re.IGNORECASE,
        ),
        "MiB",
        QuantityDimension.FILE_SIZE,
    ),
    (
        re.compile(
            r"(?:MB|megabytes?|Mo)",
            re.IGNORECASE,
        ),
        "MB",
        QuantityDimension.FILE_SIZE,
    ),

    # Power
    (
        re.compile(
            r"(?:MW|megawatts?)",
            re.IGNORECASE,
        ),
        "MW",
        QuantityDimension.POWER,
    ),
    (
        re.compile(
            r"(?:kW|kilowatts?)",
            re.IGNORECASE,
        ),
        "kW",
        QuantityDimension.POWER,
    ),
    (
        re.compile(
            r"(?:W|watts?)",
            re.IGNORECASE,
        ),
        "W",
        QuantityDimension.POWER,
    ),

    # Money
    (
        re.compile(
            r"(?:USD|US\s*dollars?|dollars?|\$)",
            re.IGNORECASE,
        ),
        "USD",
        QuantityDimension.MONEY,
    ),

    # Percentages
    (
        re.compile(
            r"(?:%|percent|pourcent)",
            re.IGNORECASE,
        ),
        "%",
        QuantityDimension.PERCENT,
    ),
)


# quantulum3 uses canonical unit names such as "watt", "kilowatt",
# "gigabyte", etc.  Map only the domain-relevant names here.
#
# Unknown quantulum units are still preserved as UNKNOWN dimension so the
# scanner does not make a semantic field decision.
_QUANTULUM_UNIT_NAME_RULES: Tuple[
    Tuple[re.Pattern[str], str, QuantityDimension],
    ...
] = (
    (
        re.compile(r"(?:gigabyte per second|gigabyte/second)", re.IGNORECASE),
        "GB/s",
        QuantityDimension.THROUGHPUT,
    ),
    (
        re.compile(r"(?:gigabit per second|gigabit/second)", re.IGNORECASE),
        "Gbps",
        QuantityDimension.THROUGHPUT,
    ),
    (
        re.compile(r"(?:megabyte per second|megabyte/second)", re.IGNORECASE),
        "MB/s",
        QuantityDimension.THROUGHPUT,
    ),
    (
        re.compile(r"(?:megabit per second|megabit/second)", re.IGNORECASE),
        "Mbps",
        QuantityDimension.THROUGHPUT,
    ),
    (
        re.compile(r"tebibyte", re.IGNORECASE),
        "TiB",
        QuantityDimension.CAPACITY,
    ),
    (
        re.compile(r"terabyte", re.IGNORECASE),
        "TB",
        QuantityDimension.CAPACITY,
    ),
    (
        re.compile(r"gibibyte", re.IGNORECASE),
        "GiB",
        QuantityDimension.FILE_SIZE,
    ),
    (
        re.compile(r"gigabyte", re.IGNORECASE),
        "GB",
        QuantityDimension.FILE_SIZE,
    ),
    (
        re.compile(r"mebibyte", re.IGNORECASE),
        "MiB",
        QuantityDimension.FILE_SIZE,
    ),
    (
        re.compile(r"megabyte", re.IGNORECASE),
        "MB",
        QuantityDimension.FILE_SIZE,
    ),
    (
        re.compile(r"megawatt", re.IGNORECASE),
        "MW",
        QuantityDimension.POWER,
    ),
    (
        re.compile(r"kilowatt", re.IGNORECASE),
        "kW",
        QuantityDimension.POWER,
    ),
    (
        re.compile(r"watt", re.IGNORECASE),
        "W",
        QuantityDimension.POWER,
    ),
    (
        re.compile(
            r"(?:united states dollar|us dollar|dollar)",
            re.IGNORECASE,
        ),
        "USD",
        QuantityDimension.MONEY,
    ),
    (
        re.compile(
            r"(?:percent|percentage|percentage point)",
            re.IGNORECASE,
        ),
        "%",
        QuantityDimension.PERCENT,
    ),
)


# Build one suffix-unit regex from the same vocabulary.
_SUFFIX_UNIT_PATTERN = (
    r"(?:"
    r"GB/s|GBps|GBPS|gb/s|gbs|Go/s|"
    r"Gbps|Gbit/s|Gbits/s|"
    r"MB/s|MBps|MBPS|mb/s|mbs|Mo/s|"
    r"Mbps|Mbit/s|Mbits/s|"
    r"TiB|tebibytes?|tebioctets?|"
    r"TB|terabytes?|t[ée]raoctets?|"
    r"GiB|gibibytes?|"
    r"GB|gigabytes?|giga(?:s)?|Go|"
    r"MiB|mebibytes?|"
    r"MB|megabytes?|Mo|"
    r"MW|megawatts?|"
    r"kW|kilowatts?|"
    r"W|watts?|"
    r"USD|US\s*dollars?|dollars?|\$|"
    r"%|percent|pourcent"
    r")"
)


_SUFFIX_QUANTITY_RE = re.compile(
    rf"(?<![\w.])"
    rf"(?P<number>{_NUMBER_PATTERN})"
    rf"(?:\s*(?P<scale>{_SCALE_PATTERN}))?"
    rf"\s*"
    rf"(?P<unit>{_SUFFIX_UNIT_PATTERN})"
    rf"(?![\w/])",
    re.IGNORECASE,
)


_PREFIX_MONEY_RE = re.compile(
    rf"(?<!\w)"
    rf"(?P<unit>US\s*\$|\$)"
    rf"\s*"
    rf"(?P<number>{_NUMBER_PATTERN})"
    rf"(?:\s*(?P<scale>{_SCALE_PATTERN}))?"
    rf"(?!\w)",
    re.IGNORECASE,
)


_BARE_NUMBER_RE = re.compile(
    rf"(?<![\w.])"
    rf"(?P<number>{_NUMBER_PATTERN})"
    rf"(?:\s*(?P<scale>{_SCALE_PATTERN}))?"
    rf"(?!\w)",
    re.IGNORECASE,
)


# =====================================================================
# STRUCTURAL UNIT RECOVERY
# =====================================================================


# Match only a known domain unit immediately after an already detected
# expression. This is used for fuzzy written-number recovery such as:
#
#     "five hunderd TiB"
#     "eight hunderd GB"
#
# where the corrected number group can stop before the symbolic unit.
_TRAILING_UNIT_RE = re.compile(
    rf"\s*(?P<unit>{_SUFFIX_UNIT_PATTERN})(?![\w/])",
    re.IGNORECASE,
)


# Deliberately narrow connector for shared-unit alternatives:
#
#     20 or 30 GB/s
#     800 ou 1200 W
#
# We do not propagate across "and", commas, arbitrary words, etc.
_ALTERNATIVE_CONNECTOR_RE = re.compile(
    r"^\s*(?:or|ou)\s*$",
    re.IGNORECASE,
)


# =====================================================================
# HELPERS
# =====================================================================


def _classify_unit(raw_unit: str) -> Tuple[str, QuantityDimension]:
    cleaned = re.sub(r"\s+", " ", raw_unit.strip())

    if cleaned.upper().replace(" ", "") == "US$":
        return "USD", QuantityDimension.MONEY

    for pattern, canonical_unit, dimension in _UNIT_RULES:
        if pattern.fullmatch(cleaned):
            return canonical_unit, dimension

    return cleaned, QuantityDimension.UNKNOWN


def _classify_quantulum_unit(
    unit_name: Optional[str],
) -> Tuple[Optional[str], QuantityDimension]:
    if not unit_name:
        return None, QuantityDimension.UNKNOWN

    cleaned = re.sub(r"\s+", " ", unit_name.strip())

    if cleaned.lower() == "dimensionless":
        return None, QuantityDimension.UNKNOWN

    for pattern, canonical_unit, dimension in _QUANTULUM_UNIT_NAME_RULES:
        if pattern.fullmatch(cleaned):
            return canonical_unit, dimension

    return cleaned, QuantityDimension.UNKNOWN


def _overlaps(
    start: int,
    end: int,
    occupied: List[Tuple[int, int]],
) -> bool:
    return any(
        start < other_end and end > other_start
        for other_start, other_end in occupied
    )


def _detection_for_digit(unit: Optional[str]) -> QuantityDetection:
    if unit is None:
        return QuantityDetection.DIGIT
    return QuantityDetection.DIGIT_WITH_UNIT


def _detection_for_quantulum(
    raw: str,
    unit: Optional[str],
) -> QuantityDetection:
    # quantulum3 is used here primarily for spelled-out numbers. If the
    # surface contains digits, keep the provenance truthful.
    contains_digit = bool(re.search(r"\d", raw))

    if contains_digit:
        return _detection_for_digit(unit)

    if unit is None:
        return QuantityDetection.NUMBER_WORDS

    return QuantityDetection.NUMBER_WORDS_WITH_UNIT



# =====================================================================
# CONTROLLED FUZZY NUMBER-WORD RECOVERY
# =====================================================================


_NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
    "point",
}

# Unit words are intentionally restricted to the project's numerical domain.
# They are used only as local plausibility anchors and optional typo targets.
_UNIT_WORDS = {
    "watt",
    "watts",
    "kilowatt",
    "kilowatts",
    "megawatt",
    "megawatts",
    "terabyte",
    "terabytes",
    "tebibyte",
    "tebibytes",
    "gigabyte",
    "gigabytes",
    "gibibyte",
    "gibibytes",
    "megabyte",
    "megabytes",
    "percent",
    "dollar",
    "dollars",
}

_FUZZY_DICTIONARY_WORDS = tuple(
    sorted(_NUMBER_WORDS | _UNIT_WORDS)
)

# Common natural-language words that are dangerously close to number words.
# The supervisor explicitly called out "for" -> "four"; these words are never
# auto-corrected by the quantity scanner.
_AMBIGUOUS_FUZZY_BLOCKLIST = {
    "for",
    "to",
    "too",
    "won",
    "there",
    "tree",

    # Common storage-domain words that are one edit away from a number
    # word and must never be auto-corrected by the QuantityScanner.
    #
    # In particular:
    #     file -> five
    #
    # A phrase such as "one file" contains a real quantity ("one"), but
    # "file" itself is semantic context, not a typo of "five".
    "file",
    "files",
}

_WORD_TOKEN_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ]+"
)

_SENTENCE_BREAK_RE = re.compile(
    r"[.!?;:\n]"
)


@dataclass(frozen=True)
class _WordToken:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _Replacement:
    start: int
    end: int
    original: str
    corrected: str


@dataclass(frozen=True)
class _MappedSegment:
    corrected_start: int
    corrected_end: int
    original_start: int
    original_end: int
    was_replaced: bool


def _build_symspell() -> SymSpell:
    """
    Build an in-memory SymSpell dictionary containing only controlled
    number-related words and known unit words.
    """

    symspell = SymSpell(
        max_dictionary_edit_distance=1,
        prefix_length=7,
    )

    for word in _FUZZY_DICTIONARY_WORDS:
        symspell.create_dictionary_entry(word, 1)

    return symspell


def _tokenize_words(text: str) -> List[_WordToken]:
    return [
        _WordToken(
            text=match.group(0),
            start=match.start(),
            end=match.end(),
        )
        for match in _WORD_TOKEN_RE.finditer(text)
    ]


def _same_local_expression(
    text: str,
    first: _WordToken,
    second: _WordToken,
) -> bool:
    """
    Conservative local-context test.

    Fuzzy correction is allowed only when a suspicious token is close to an
    exact number/unit word without crossing a strong sentence boundary.
    """

    left = min(first.end, second.end)
    right = max(first.start, second.start)

    if right < left:
        between = ""
    else:
        between = text[left:right]

    if _SENTENCE_BREAK_RE.search(between):
        return False

    return len(between) <= 20


def _lookup_fuzzy_candidate(
    symspell: SymSpell,
    token: str,
) -> Optional[str]:
    """
    Return one controlled fuzzy correction candidate.

    Important details
    -----------------
    - only edit distance 1 is accepted;
    - ordinary ambiguous words remain blocked;
    - when several controlled words are equally close, preserve simple
      morphology when possible (especially plural ``s``).

    Example:
        kilowats
        -> both "kilowatt" and "kilowatts" can be close candidates
        -> original ends with "s"
        -> prefer "kilowatts"
    """

    lowered = token.lower()

    if lowered in _AMBIGUOUS_FUZZY_BLOCKLIST:
        return None

    if lowered in _NUMBER_WORDS or lowered in _UNIT_WORDS:
        return None

    suggestions = symspell.lookup(
        lowered,
        Verbosity.ALL,
        max_edit_distance=1,
        include_unknown=False,
    )

    valid = [
        suggestion
        for suggestion in suggestions
        if suggestion.distance == 1
        and (
            suggestion.term in _NUMBER_WORDS
            or suggestion.term in _UNIT_WORDS
        )
    ]

    if not valid:
        return None

    original_is_plural = lowered.endswith("s")

    valid.sort(
        key=lambda suggestion: (
            suggestion.distance,
            0
            if suggestion.term.endswith("s") == original_is_plural
            else 1,
            abs(len(suggestion.term) - len(lowered)),
            suggestion.term,
        )
    )

    return valid[0].term


def _find_controlled_replacements(
    text: str,
    symspell: SymSpell,
) -> List[_Replacement]:
    """
    Find typo corrections only inside a contiguous plausible numerical
    expression.

    Why adjacency matters
    ---------------------
    A previous implementation accepted fuzzy tokens that were merely within
    three words of an exact number/unit word. That could incorrectly turn:

        "Average file size is two gigabtyes"

    into a false extra number:

        file -> five

    because ``file`` is edit-distance 1 from ``five`` and happens to be near
    ``two``.

    The safer rule is to grow a numerical expression from exact controlled
    anchors only through ADJACENT controlled/fuzzy tokens.

    This still accepts:
        two hunderd
        eight hunderd wats
        two hundred gigabtyes

    while rejecting unrelated nearby words such as:
        file size is two ...
    """

    tokens = _tokenize_words(text)

    if not tokens:
        return []

    exact_anchor_indices = {
        index
        for index, token in enumerate(tokens)
        if token.text.lower() in _NUMBER_WORDS
        or token.text.lower() in _UNIT_WORDS
    }

    if not exact_anchor_indices:
        return []

    # Pre-compute controlled fuzzy candidates once.
    fuzzy_candidates = {
        index: candidate
        for index, token in enumerate(tokens)
        if (
            candidate := _lookup_fuzzy_candidate(
                symspell,
                token.text,
            )
        )
        is not None
    }

    # Accepted indices initially contain only exact controlled words.
    # Then expand iteratively to directly adjacent fuzzy candidates.
    accepted_indices = set(exact_anchor_indices)
    accepted_fuzzy_indices = set()

    changed = True

    while changed:
        changed = False

        for index, candidate in fuzzy_candidates.items():
            if index in accepted_indices:
                continue

            neighbours = [
                neighbour
                for neighbour in (index - 1, index + 1)
                if neighbour in accepted_indices
                and 0 <= neighbour < len(tokens)
            ]

            if not neighbours:
                continue

            token = tokens[index]

            if not any(
                _same_local_expression(
                    text,
                    token,
                    tokens[neighbour],
                )
                for neighbour in neighbours
            ):
                continue

            accepted_indices.add(index)
            accepted_fuzzy_indices.add(index)
            changed = True

    replacements: List[_Replacement] = []

    for index in sorted(accepted_fuzzy_indices):
        token = tokens[index]
        candidate = fuzzy_candidates[index]

        replacements.append(
            _Replacement(
                start=token.start,
                end=token.end,
                original=token.text,
                corrected=candidate,
            )
        )

    return replacements


def _apply_replacements_with_mapping(
    text: str,
    replacements: List[_Replacement],
) -> Tuple[str, List[_MappedSegment]]:
    """
    Apply only approved token replacements and retain a mapping back to the
    original source offsets.
    """

    if not replacements:
        return text, [
            _MappedSegment(
                corrected_start=0,
                corrected_end=len(text),
                original_start=0,
                original_end=len(text),
                was_replaced=False,
            )
        ]

    replacements = sorted(
        replacements,
        key=lambda item: item.start,
    )

    corrected_parts: List[str] = []
    segments: List[_MappedSegment] = []

    original_cursor = 0
    corrected_cursor = 0

    for replacement in replacements:
        if replacement.start < original_cursor:
            continue

        if replacement.start > original_cursor:
            unchanged = text[
                original_cursor:replacement.start
            ]
            corrected_parts.append(unchanged)

            segments.append(
                _MappedSegment(
                    corrected_start=corrected_cursor,
                    corrected_end=corrected_cursor + len(unchanged),
                    original_start=original_cursor,
                    original_end=replacement.start,
                    was_replaced=False,
                )
            )

            corrected_cursor += len(unchanged)

        corrected_parts.append(replacement.corrected)

        segments.append(
            _MappedSegment(
                corrected_start=corrected_cursor,
                corrected_end=corrected_cursor + len(replacement.corrected),
                original_start=replacement.start,
                original_end=replacement.end,
                was_replaced=True,
            )
        )

        corrected_cursor += len(replacement.corrected)
        original_cursor = replacement.end

    if original_cursor < len(text):
        unchanged = text[original_cursor:]
        corrected_parts.append(unchanged)

        segments.append(
            _MappedSegment(
                corrected_start=corrected_cursor,
                corrected_end=corrected_cursor + len(unchanged),
                original_start=original_cursor,
                original_end=len(text),
                was_replaced=False,
            )
        )

    return "".join(corrected_parts), segments


def _map_corrected_span_to_original(
    corrected_start: int,
    corrected_end: int,
    segments: List[_MappedSegment],
) -> Optional[Tuple[int, int, bool]]:
    overlapping = [
        segment
        for segment in segments
        if corrected_start < segment.corrected_end
        and corrected_end > segment.corrected_start
    ]

    if not overlapping:
        return None

    original_start = min(
        segment.original_start
        for segment in overlapping
    )
    original_end = max(
        segment.original_end
        for segment in overlapping
    )
    used_replacement = any(
        segment.was_replaced
        for segment in overlapping
    )

    return original_start, original_end, used_replacement


def _build_fuzzy_candidate_groups(
    text: str,
    replacements: List[_Replacement],
) -> List[Tuple[int, int, str, str]]:
    """
    Build minimal local numerical-expression spans around approved fuzzy
    replacements.

    This deliberately avoids sending the complete corrected sentence to
    quantulum3. Some natural-language parsers may return a span that contains
    surrounding words (for example, "We have two hundred"). The scanner must
    preserve only the quantity evidence itself.

    A candidate group is made only of consecutive controlled tokens:
    - exact number words;
    - exact known unit words;
    - approved fuzzy replacements of those words.

    Returns tuples:
        (original_start, original_end, raw, normalized)
    """

    if not replacements:
        return []

    tokens = _tokenize_words(text)
    replacement_by_span = {
        (replacement.start, replacement.end): replacement
        for replacement in replacements
    }

    eligible: List[Tuple[int, _WordToken, str, bool]] = []

    for index, token in enumerate(tokens):
        lowered = token.text.lower()
        replacement = replacement_by_span.get(
            (token.start, token.end)
        )

        if replacement is not None:
            eligible.append(
                (index, token, replacement.corrected, True)
            )
            continue

        if lowered in _NUMBER_WORDS or lowered in _UNIT_WORDS:
            eligible.append(
                (index, token, token.text, False)
            )

    if not eligible:
        return []

    groups: List[List[Tuple[int, _WordToken, str, bool]]] = []
    current: List[Tuple[int, _WordToken, str, bool]] = []

    for item in eligible:
        index, token, corrected, was_replaced = item

        if not current:
            current = [item]
            continue

        previous_index, previous_token, _, _ = current[-1]
        between = text[previous_token.end:token.start]

        # Keep only genuinely local expression pieces together. A normal
        # lexical word between two controlled tokens ends the expression.
        contiguous_token = index == previous_index + 1
        harmless_separator = bool(
            re.fullmatch(r"[\s\-]*", between)
        )

        if contiguous_token and harmless_separator:
            current.append(item)
        else:
            groups.append(current)
            current = [item]

    if current:
        groups.append(current)

    output: List[Tuple[int, int, str, str]] = []

    for group in groups:
        if not any(item[3] for item in group):
            continue

        corrected_words = [item[2].lower() for item in group]

        # The expression must actually contain numerical content. A fuzzy
        # unit alone is not enough to create a quantity.
        has_number_word = any(
            word in _NUMBER_WORDS
            for word in corrected_words
        )

        if not has_number_word:
            continue

        original_start = group[0][1].start
        original_end = group[-1][1].end
        raw = text[original_start:original_end]

        normalized_parts: List[str] = []
        cursor = original_start

        for _, token, corrected, _ in group:
            if token.start > cursor:
                normalized_parts.append(
                    text[cursor:token.start]
                )

            normalized_parts.append(corrected)
            cursor = token.end

        if cursor < original_end:
            normalized_parts.append(
                text[cursor:original_end]
            )

        normalized = "".join(normalized_parts)

        output.append(
            (
                original_start,
                original_end,
                raw,
                normalized,
            )
        )

    return output



def _recover_trailing_unit_after_fuzzy_expression(
    text: str,
    original_start: int,
    original_end: int,
    raw: str,
    normalized: str,
    canonical_unit: Optional[str],
    dimension: QuantityDimension,
) -> Tuple[
    int,
    str,
    str,
    Optional[str],
    QuantityDimension,
]:
    """
    Recover an explicit unit written immediately after a fuzzy-corrected
    number expression.

    Example
    -------
    Original:
        "five hunderd TiB"

    Fuzzy number group:
        raw        = "five hunderd"
        normalized = "five hundred"

    The numerical value can be correctly recovered as 500 while the local
    fuzzy group stops before the symbolic unit "TiB". This helper safely
    extends the detected surface to include that explicit trailing unit.

    Result:
        raw        = "five hunderd TiB"
        normalized = "five hundred TiB"
        unit       = "TiB"
        dimension  = CAPACITY

    Safety
    ------
    - no unit is invented;
    - only an explicitly present, known domain unit is attached;
    - if quantulum3 already returned a unit, nothing is changed.
    """

    if canonical_unit is not None:
        return (
            original_end,
            raw,
            normalized,
            canonical_unit,
            dimension,
        )

    suffix = text[original_end:]

    match = _TRAILING_UNIT_RE.match(suffix)

    if match is None:
        return (
            original_end,
            raw,
            normalized,
            canonical_unit,
            dimension,
        )

    raw_unit = match.group("unit")

    recovered_unit, recovered_dimension = _classify_unit(raw_unit)

    if recovered_dimension == QuantityDimension.UNKNOWN:
        return (
            original_end,
            raw,
            normalized,
            canonical_unit,
            dimension,
        )

    final_end = original_end + match.end()

    final_raw = text[original_start:final_end]

    # Preserve the corrected number words while appending the exact source
    # suffix (spacing + unit spelling) after them.
    final_normalized = normalized + suffix[:match.end()]

    return (
        final_end,
        final_raw,
        final_normalized,
        recovered_unit,
        recovered_dimension,
    )


def _propagate_shared_unit_to_alternatives(
    text: str,
    detected: List[_DetectedQuantity],
) -> List[_DetectedQuantity]:
    """
    Propagate a trailing explicit unit across a narrow alternative syntax.

    Examples
    --------
    "20 or 30 GB/s"

        before:
            20, unit=None
            30, unit=GB/s

        after:
            20, unit=GB/s
            30, unit=GB/s

    "800 ou 1200 W"

        before:
            800, unit=None
            1200, unit=W

        after:
            800, unit=W
            1200, unit=W

    Safety
    ------
    Propagation occurs only when:
    - quantities are adjacent in source order;
    - the left quantity is unitless and UNKNOWN-dimension;
    - the right quantity has a known unit/dimension;
    - the text between them is only "or" or "ou".

    This does not resolve the ambiguity and does not choose one value.
    It only preserves the syntactically shared unit.
    """

    if len(detected) < 2:
        return detected

    output = list(detected)

    # Right-to-left also supports chains such as:
    #     20 or 30 or 40 GB/s
    for index in range(len(output) - 2, -1, -1):
        left = output[index]
        right = output[index + 1]

        if left.unit is not None:
            continue

        if left.dimension != QuantityDimension.UNKNOWN:
            continue

        if right.unit is None:
            continue

        if right.dimension == QuantityDimension.UNKNOWN:
            continue

        between = text[left.end:right.start]

        if not _ALTERNATIVE_CONNECTOR_RE.fullmatch(between):
            continue

        output[index] = replace(
            left,
            unit=right.unit,
            dimension=right.dimension,
        )

    return output


def _detect_fuzzy_quantities(
    text: str,
    occupied: List[Tuple[int, int]],
    symspell: SymSpell,
) -> List[_DetectedQuantity]:
    """
    Detect numerical expressions requiring controlled typo recovery.

    Flow
    ----
    original text
        -> controlled SymSpell replacements
        -> minimal fuzzy numerical groups
        -> quantulum3 numerical parsing
        -> explicit trailing-unit recovery
        -> _DetectedQuantity

    Example:
        "five hunderd TiB"

        hunderd -> hundred
        five hundred -> 500
        trailing TiB -> unit=TiB, dimension=CAPACITY
    """

    replacements = _find_controlled_replacements(
        text,
        symspell,
    )

    if not replacements:
        return []

    candidate_groups = _build_fuzzy_candidate_groups(
        text,
        replacements,
    )

    output: List[_DetectedQuantity] = []

    for original_start, original_end, raw, normalized in candidate_groups:
        if _overlaps(
            original_start,
            original_end,
            occupied,
        ):
            continue

        # -------------------------------------------------------------
        # 1. Parse the corrected local numerical expression
        # -------------------------------------------------------------
        try:
            parsed_items = quantulum_parser.parse(normalized)
        except Exception:
            continue

        if not parsed_items:
            continue

        # Candidate groups are deliberately minimal. If quantulum3 returns
        # more than one parse, keep the parse explaining the largest span.
        parsed = max(
            parsed_items,
            key=lambda item: (
                int(item.span[1]) - int(item.span[0])
                if getattr(item, "span", None)
                else 0
            ),
        )

        # -------------------------------------------------------------
        # 2. Numerical value
        # -------------------------------------------------------------
        try:
            value = _to_python_number(parsed.value)
        except (TypeError, ValueError, InvalidOperation):
            continue

        # -------------------------------------------------------------
        # 3. Unit detected directly by quantulum3
        # -------------------------------------------------------------
        unit_name = getattr(
            getattr(parsed, "unit", None),
            "name",
            None,
        )

        canonical_unit, dimension = _classify_quantulum_unit(
            unit_name
        )

        # -------------------------------------------------------------
        # 4. Recover an explicit symbolic unit immediately following
        #    the fuzzy number group.
        #
        #    Examples:
        #       five hunderd TiB
        #       eight hunderd GB
        # -------------------------------------------------------------
        (
            final_end,
            final_raw,
            final_normalized,
            canonical_unit,
            dimension,
        ) = _recover_trailing_unit_after_fuzzy_expression(
            text=text,
            original_start=original_start,
            original_end=original_end,
            raw=raw,
            normalized=normalized,
            canonical_unit=canonical_unit,
            dimension=dimension,
        )

        # The recovered unit can extend the original fuzzy span, therefore
        # validate overlap again using the complete final surface.
        if _overlaps(
            original_start,
            final_end,
            occupied,
        ):
            continue

        detection = (
            QuantityDetection.FUZZY_NUMBER_WORDS
            if canonical_unit is None
            else QuantityDetection.FUZZY_NUMBER_WORDS_WITH_UNIT
        )

        output.append(
            _DetectedQuantity(
                raw=final_raw,
                normalized=final_normalized,
                value=value,
                unit=canonical_unit,
                dimension=dimension,
                start=original_start,
                end=final_end,
                detection=detection,
                corrected=True,
            )
        )

    return output


# =====================================================================
# PUBLIC SCANNER
# =====================================================================


class QuantityScanner:
    """
    Quantity detector for the selective extraction cascade.

    Detection strategy
    ------------------
    1. Preserve the existing deterministic digit/known-unit rules.
    2. Use quantulum3 for non-overlapping quantities that the direct
       rules did not capture, especially written numbers.

    Typo recovery is handled conservatively with symspellpy using a
    restricted in-memory dictionary. Only tokens that belong to a plausible
    local number expression can be corrected.

    Responsibilities
    ----------------
    - detect quantities;
    - preserve exact source spans;
    - keep raw and normalized forms;
    - parse the numeric value;
    - identify units/dimensions when possible;
    - record how the quantity was detected.

    Non-responsibilities
    --------------------
    - no final requirement field selection;
    - no semantic role inference;
    - no business validation;
    - no LLM/embedding usage.
    """

    def __init__(self) -> None:
        self._symspell = _build_symspell()

    def scan(self, text: str) -> List[Quantity]:
        if not text:
            return []

        detected: List[_DetectedQuantity] = []
        occupied: List[Tuple[int, int]] = []

        # -------------------------------------------------------------
        # 1. Prefix currency forms: "$100000", "US$ 100000"
        # -------------------------------------------------------------
        for match in _PREFIX_MONEY_RE.finditer(text):
            start, end = match.span()

            if _overlaps(start, end, occupied):
                continue

            value = _parse_number(
                match.group("number"),
                match.group("scale"),
            )

            raw = text[start:end]

            detected.append(
                _DetectedQuantity(
                    raw=raw,
                    normalized=raw,
                    value=value,
                    unit="USD",
                    dimension=QuantityDimension.MONEY,
                    start=start,
                    end=end,
                    detection=QuantityDetection.DIGIT_WITH_UNIT,
                )
            )
            occupied.append((start, end))

        # -------------------------------------------------------------
        # 2. Explicit suffix units: "800 W", "500 TiB", "70 %", ...
        # -------------------------------------------------------------
        for match in _SUFFIX_QUANTITY_RE.finditer(text):
            start, end = match.span()

            if _overlaps(start, end, occupied):
                continue

            value = _parse_number(
                match.group("number"),
                match.group("scale"),
            )

            canonical_unit, dimension = _classify_unit(
                match.group("unit")
            )

            raw = text[start:end]

            detected.append(
                _DetectedQuantity(
                    raw=raw,
                    normalized=raw,
                    value=value,
                    unit=canonical_unit,
                    dimension=dimension,
                    start=start,
                    end=end,
                    detection=QuantityDetection.DIGIT_WITH_UNIT,
                )
            )
            occupied.append((start, end))

        # -------------------------------------------------------------
        # 3. Unitless digit quantities: "200", "10 million", ...
        # -------------------------------------------------------------
        for match in _BARE_NUMBER_RE.finditer(text):
            start, end = match.span()

            if _overlaps(start, end, occupied):
                continue

            value = _parse_number(
                match.group("number"),
                match.group("scale"),
            )

            raw = text[start:end]

            detected.append(
                _DetectedQuantity(
                    raw=raw,
                    normalized=raw,
                    value=value,
                    unit=None,
                    dimension=QuantityDimension.UNKNOWN,
                    start=start,
                    end=end,
                    detection=QuantityDetection.DIGIT,
                )
            )
            occupied.append((start, end))

        # -------------------------------------------------------------
        # 4. Controlled SymSpell recovery before the normal quantulum pass.
        #
        # This ordering is important for inputs such as "two hunderd":
        # quantulum3 might otherwise recognize only the valid fragment "two".
        # The fuzzy pass first repairs the plausible full number expression
        # and reserves its original source span.
        # -------------------------------------------------------------
        fuzzy_items = _detect_fuzzy_quantities(
            text=text,
            occupied=occupied,
            symspell=self._symspell,
        )

        for item in fuzzy_items:
            detected.append(item)
            occupied.append((item.start, item.end))

        # -------------------------------------------------------------
        # 5. quantulum3 fallback for written numbers / other quantities
        # -------------------------------------------------------------
        #
        # quantulum3 exposes:
        # - surface: exact matched surface
        # - span: (start, end)
        # - value: parsed numeric value
        # - unit.name: parsed unit name
        #
        # Existing direct detections win on overlap so we preserve the
        # project's established canonical unit behavior for digit forms.
        try:
            quantulum_quantities = quantulum_parser.parse(text)
        except Exception:
            # Degrade safely: direct deterministic scanning remains
            # available even if quantulum3 cannot parse a particular text.
            quantulum_quantities = []

        for parsed in quantulum_quantities:
            try:
                start, end = parsed.span
                start = int(start)
                end = int(end)
            except (TypeError, ValueError, AttributeError):
                continue

            if start < 0 or end <= start or end > len(text):
                continue

            if _overlaps(start, end, occupied):
                continue

            raw = text[start:end]

            try:
                value = _to_python_number(parsed.value)
            except (TypeError, ValueError, InvalidOperation):
                continue

            unit_name = getattr(
                getattr(parsed, "unit", None),
                "name",
                None,
            )

            canonical_unit, dimension = _classify_quantulum_unit(
                unit_name
            )

            detection = _detection_for_quantulum(
                raw=raw,
                unit=canonical_unit,
            )

            detected.append(
                _DetectedQuantity(
                    raw=raw,
                    normalized=raw,
                    value=value,
                    unit=canonical_unit,
                    dimension=dimension,
                    start=start,
                    end=end,
                    detection=detection,
                    corrected=False,
                )
            )
            occupied.append((start, end))

        # -------------------------------------------------------------
        # 6. Preserve source order
        # -------------------------------------------------------------
        detected.sort(
            key=lambda item: (
                item.start,
                item.end,
            )
        )

        # -------------------------------------------------------------
        # 7. Propagate an explicitly shared trailing unit across narrow
        #    alternatives such as:
        #
        #       20 or 30 GB/s
        #       800 ou 1200 W
        #
        #    Both values remain present; the scanner does not choose one.
        # -------------------------------------------------------------
        detected = _propagate_shared_unit_to_alternatives(
            text,
            detected,
        )

        # -------------------------------------------------------------
        # 8. Public Quantity objects with stable q1/q2/... identifiers
        # -------------------------------------------------------------
        return [
            Quantity(
                id=f"q{index}",
                raw=item.raw,
                normalized=item.normalized,
                value=item.value,
                unit=item.unit,
                dimension=item.dimension,
                start=item.start,
                end=item.end,
                source_text=text,
                detection=item.detection,
                corrected=item.corrected,
            )
            for index, item in enumerate(detected, start=1)
        ]

    def mark_quantities(
        self,
        text: str,
        quantities: Optional[List[Quantity]] = None,
    ) -> str:
        """
        Insert [Q]...[/Q] around detected quantities.

        Example:
            "Anything above eight hundred watts would overload the supply."

        becomes:
            "Anything above [Q]eight hundred watts[/Q] would overload the supply."

        When several quantities exist, every detected quantity is marked.
        """

        items = quantities if quantities is not None else self.scan(text)

        if not items:
            return text

        output = text

        # Insert from the end so that earlier offsets remain valid.
        for item in sorted(items, key=lambda q: q.start, reverse=True):
            output = (
                output[:item.start]
                + "[Q]"
                + output[item.start:item.end]
                + "[/Q]"
                + output[item.end:]
            )

        return output