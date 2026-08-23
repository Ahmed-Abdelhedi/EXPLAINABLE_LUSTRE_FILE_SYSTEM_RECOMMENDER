from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

from .models import Quantity, QuantityDetection, QuantityDimension
from .quantity_scanner import QuantityScanner


# =====================================================================
# CONTROLLED ROBUSTNESS VOCABULARY
# =====================================================================
#
# This layer is intentionally conservative.  It does NOT perform arbitrary
# spell correction on the user's sentence.  It only repairs a numerical
# expression when there is strong local evidence:
#
#   1) a digit quantity is immediately followed by a one-edit typo of a
#      known domain unit, e.g. ``800 wats`` -> ``800 watts``; or
#   2) one/more English number words immediately precede an EXPLICIT known
#      unit and exactly one number token needs a one-edit correction, e.g.
#      ``fourty GB`` -> ``forty GB``.
#
# The corrected text is used only for parsing.  ``raw`` and source offsets
# always refer to the original user message.
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

# Ordinary words that are dangerously close to number words and must never be
# auto-corrected into quantities.
_NUMBER_TYPO_BLOCKLIST = {
    "for",   # for -> four
    "to",
    "too",
    "won",
    "there",
    "tree",
    "file",  # file -> five
    "files",
}

# Fuzzy unit recovery after an already detected digit quantity.
# Only spelled-out units are fuzzy-corrected; symbolic abbreviations such as
# W/GB/GB/s are already handled by the base QuantityScanner.
_UNIT_WORD_TO_CANONICAL = {
    "watt": ("W", QuantityDimension.POWER),
    "watts": ("W", QuantityDimension.POWER),
    "kilowatt": ("kW", QuantityDimension.POWER),
    "kilowatts": ("kW", QuantityDimension.POWER),
    "megawatt": ("MW", QuantityDimension.POWER),
    "megawatts": ("MW", QuantityDimension.POWER),
    "gigabyte": ("GB", QuantityDimension.FILE_SIZE),
    "gigabytes": ("GB", QuantityDimension.FILE_SIZE),
    "gibibyte": ("GiB", QuantityDimension.FILE_SIZE),
    "gibibytes": ("GiB", QuantityDimension.FILE_SIZE),
    "megabyte": ("MB", QuantityDimension.FILE_SIZE),
    "megabytes": ("MB", QuantityDimension.FILE_SIZE),
    "tebibyte": ("TiB", QuantityDimension.CAPACITY),
    "tebibytes": ("TiB", QuantityDimension.CAPACITY),
    "terabyte": ("TB", QuantityDimension.CAPACITY),
    "terabytes": ("TB", QuantityDimension.CAPACITY),
    "dollar": ("USD", QuantityDimension.MONEY),
    "dollars": ("USD", QuantityDimension.MONEY),
    "percent": ("%", QuantityDimension.PERCENT),
}

# Explicit units that may anchor recovery of a misspelled written number.
# Longer/more specific forms must come first.
_EXPLICIT_UNIT_RE = re.compile(
    r"(?<![\w/])"
    r"(?:"
    r"GB/s|GBps|GBPS|Gbps|Gbit/s|Gbits/s|"
    r"MB/s|MBps|MBPS|Mbps|Mbit/s|Mbits/s|"
    r"TiB|TB|GiB|GB|MiB|MB|"
    r"kW|MW|W|USD|%"
    r")"
    r"(?![\w/])",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z]+")
_IMMEDIATE_WORD_AFTER_RE = re.compile(r"\s+(?P<word>[A-Za-z]+)\b")


def _levenshtein_distance_at_most_one(left: str, right: str) -> Optional[int]:
    """Return 0/1 when edit distance <= 1, otherwise None."""

    a = left.lower()
    b = right.lower()

    if a == b:
        return 0

    if abs(len(a) - len(b)) > 1:
        return None

    # Same length: at most one substitution.
    if len(a) == len(b):
        mismatches = sum(ch1 != ch2 for ch1, ch2 in zip(a, b))
        return 1 if mismatches == 1 else None

    # Ensure a is the shorter string.
    if len(a) > len(b):
        a, b = b, a

    # One insertion/deletion.
    i = 0
    j = 0
    edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue

        edits += 1
        if edits > 1:
            return None
        j += 1

    return 1


def _unique_one_edit_candidate(token: str, vocabulary: Iterable[str]) -> Optional[str]:
    lowered = token.lower()
    candidates = [
        word
        for word in vocabulary
        if _levenshtein_distance_at_most_one(lowered, word) == 1
    ]

    if not candidates:
        return None

    # Preserve simple plural morphology when possible.
    plural = lowered.endswith("s")
    candidates.sort(
        key=lambda word: (
            0 if word.endswith("s") == plural else 1,
            abs(len(word) - len(lowered)),
            word,
        )
    )

    best = candidates[0]
    # If two candidates are equally plausible on all meaningful criteria,
    # abstain instead of guessing.
    if len(candidates) > 1:
        key0 = (
            candidates[0].endswith("s") == plural,
            abs(len(candidates[0]) - len(lowered)),
        )
        key1 = (
            candidates[1].endswith("s") == plural,
            abs(len(candidates[1]) - len(lowered)),
        )
        if key0 == key1:
            return None

    return best


def _overlaps(start: int, end: int, quantities: Sequence[Quantity]) -> bool:
    return any(start < q.end and end > q.start for q in quantities)


def _renumber_in_source_order(quantities: Sequence[Quantity]) -> List[Quantity]:
    ordered = sorted(quantities, key=lambda q: (q.start, q.end))
    return [replace(quantity, id=f"q{index}") for index, quantity in enumerate(ordered, 1)]


class RobustQuantityScanner:
    """
    Safe robustness adapter around the project's existing QuantityScanner.

    Why an adapter instead of a global spell-corrector?
    ---------------------------------------------------
    The QuantityScanner must remain evidence-preserving and conservative.
    Generic spell correction can create false numerical evidence (for example
    ``for`` -> ``four`` or ``file`` -> ``five``).  This adapter therefore uses
    only narrow, structurally anchored recovery rules.

    The class exposes the same ``scan(text)`` API as QuantityScanner and can be
    injected directly into SelectiveCascade.
    """

    def __init__(self, base_scanner: Optional[QuantityScanner] = None) -> None:
        self.base_scanner = base_scanner or QuantityScanner()

    def scan(self, text: str) -> List[Quantity]:
        if not text:
            return []

        base_quantities = list(self.base_scanner.scan(text))

        # 1. Upgrade an already detected unitless digit quantity when an
        #    immediately following unit word is a unique one-edit typo.
        upgraded = self._recover_fuzzy_units_after_digits(text, base_quantities)

        # 2. Recover a misspelled written-number expression only when it is
        #    immediately anchored by an explicit known unit.
        recovered_written = self._recover_fuzzy_number_words_before_units(
            text,
            upgraded,
        )

        merged = list(upgraded)
        for quantity in recovered_written:
            if not _overlaps(quantity.start, quantity.end, merged):
                merged.append(quantity)

        return _renumber_in_source_order(merged)

    # -----------------------------------------------------------------
    # Digit + fuzzy unit, e.g. 800 wats
    # -----------------------------------------------------------------
    def _recover_fuzzy_units_after_digits(
        self,
        text: str,
        quantities: Sequence[Quantity],
    ) -> List[Quantity]:
        output: List[Quantity] = []

        for quantity in quantities:
            if quantity.unit is not None or quantity.dimension != QuantityDimension.UNKNOWN:
                output.append(quantity)
                continue

            # This recovery is intentionally restricted to quantities whose
            # original evidence contains a digit.  Written-number typos are
            # handled by the separate routine below.
            if not re.search(r"\d", quantity.raw):
                output.append(quantity)
                continue

            suffix = text[quantity.end:]
            match = _IMMEDIATE_WORD_AFTER_RE.match(suffix)
            if match is None:
                output.append(quantity)
                continue

            raw_word = match.group("word")
            corrected_word = _unique_one_edit_candidate(
                raw_word,
                _UNIT_WORD_TO_CANONICAL.keys(),
            )
            if corrected_word is None:
                output.append(quantity)
                continue

            canonical_unit, dimension = _UNIT_WORD_TO_CANONICAL[corrected_word]
            final_end = quantity.end + match.end()
            final_raw = text[quantity.start:final_end]
            normalized = (
                quantity.normalized
                + suffix[: match.start("word")]
                + corrected_word
            )

            output.append(
                replace(
                    quantity,
                    raw=final_raw,
                    normalized=normalized,
                    unit=canonical_unit,
                    dimension=dimension,
                    end=final_end,
                    detection=QuantityDetection.DIGIT_WITH_UNIT,
                    corrected=True,
                )
            )

        return output

    # -----------------------------------------------------------------
    # Fuzzy written number + explicit unit, e.g. fourty GB
    # -----------------------------------------------------------------
    def _recover_fuzzy_number_words_before_units(
        self,
        text: str,
        existing_quantities: Sequence[Quantity],
    ) -> List[Quantity]:
        recovered: List[Quantity] = []

        for unit_match in _EXPLICIT_UNIT_RE.finditer(text):
            unit_start = unit_match.start()

            # If an existing quantity already reaches this unit, no recovery
            # is required.
            if any(
                q.unit is not None
                and q.start < unit_match.end()
                and q.end >= unit_start
                for q in existing_quantities
            ):
                continue

            prefix = text[:unit_start]
            word_matches = list(_WORD_RE.finditer(prefix))
            if not word_matches:
                continue

            # The last word must be immediately adjacent to the unit modulo
            # whitespace/hyphen.  Otherwise the unit is not a safe anchor for
            # the number expression.
            last = word_matches[-1]
            between_last_and_unit = text[last.end():unit_start]
            if not re.fullmatch(r"[\s\-]*", between_last_and_unit):
                continue

            selected: List[Tuple[re.Match[str], str, bool]] = []
            corrected_count = 0

            for word_match in reversed(word_matches[-5:]):
                token = word_match.group(0)
                lowered = token.lower()

                if selected:
                    gap = text[word_match.end():selected[-1][0].start()]
                    if not re.fullmatch(r"[\s\-]*", gap):
                        break

                if lowered in _NUMBER_WORDS:
                    selected.append((word_match, lowered, False))
                    continue

                if lowered in _NUMBER_TYPO_BLOCKLIST:
                    break

                corrected = _unique_one_edit_candidate(lowered, _NUMBER_WORDS)
                if corrected is None:
                    break

                selected.append((word_match, corrected, True))
                corrected_count += 1

            if not selected or corrected_count == 0:
                continue

            selected.reverse()
            original_start = selected[0][0].start()
            original_end = unit_match.end()

            if _overlaps(original_start, original_end, existing_quantities):
                continue

            # Require at least one corrected token, but never more than one in
            # this automatic path.  Multiple simultaneous spelling errors are
            # safer to leave for a later clarification/fallback strategy.
            if corrected_count != 1:
                continue

            normalized_number_parts: List[str] = []
            cursor = original_start
            for word_match, corrected_word, _ in selected:
                if word_match.start() > cursor:
                    normalized_number_parts.append(text[cursor:word_match.start()])
                normalized_number_parts.append(corrected_word)
                cursor = word_match.end()

            normalized_number_parts.append(text[cursor:unit_match.end()])
            normalized_local = "".join(normalized_number_parts)

            # Delegate actual numerical/unit parsing to the already validated
            # base scanner.  The adapter changes only the spelling evidence,
            # not the numerical semantics.
            parsed = self.base_scanner.scan(normalized_local)
            candidates = [q for q in parsed if q.unit is not None]
            if len(candidates) != 1:
                continue

            parsed_quantity = candidates[0]
            recovered.append(
                Quantity(
                    id="pending",
                    raw=text[original_start:original_end],
                    normalized=normalized_local,
                    value=parsed_quantity.value,
                    unit=parsed_quantity.unit,
                    dimension=parsed_quantity.dimension,
                    start=original_start,
                    end=original_end,
                    source_text=text,
                    detection=QuantityDetection.FUZZY_NUMBER_WORDS_WITH_UNIT,
                    corrected=True,
                )
            )

        return recovered

    def info(self) -> dict:
        return {
            "name": "RobustQuantityScanner",
            "base": type(self.base_scanner).__name__,
            "digit_fuzzy_unit_recovery": True,
            "written_number_fuzzy_recovery": True,
            "max_auto_corrected_number_tokens_per_expression": 1,
        }
