from __future__ import annotations

import math
import re
from typing import Optional


_RATIO_SEP = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:/|:|-)\s*(\d+(?:\.\d+)?)\s*$",
    flags=re.I,
)

_READ_THEN_WRITE = re.compile(
    r"\b(?:read|lecture)\b[^\d]{0,20}"
    r"(\d+(?:\.\d+)?)\s*%?"
    r".{0,30}?"
    r"\b(?:write|ecriture|écriture)\b[^\d]{0,20}"
    r"(\d+(?:\.\d+)?)\s*%?",
    flags=re.I,
)

_NUM_READ_NUM_WRITE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%?\s*"
    r"(?:read|lecture)\b"
    r".{0,30}?"
    r"(\d+(?:\.\d+)?)\s*%?\s*"
    r"(?:write|ecriture|écriture)\b",
    flags=re.I,
)

_WRITE_THEN_READ = re.compile(
    r"\b(?:write|ecriture|écriture)\b[^\d]{0,20}"
    r"(\d+(?:\.\d+)?)\s*%?"
    r".{0,30}?"
    r"\b(?:read|lecture)\b[^\d]{0,20}"
    r"(\d+(?:\.\d+)?)\s*%?",
    flags=re.I,
)

_NUM_WRITE_NUM_READ = re.compile(
    r"(\d+(?:\.\d+)?)\s*%?\s*"
    r"(?:write|ecriture|écriture)\b"
    r".{0,30}?"
    r"(\d+(?:\.\d+)?)\s*%?\s*"
    r"(?:read|lecture)\b",
    flags=re.I,
)


def _normalize_pair(
    read_value: float,
    write_value: float,
) -> Optional[dict]:
    if not (
        math.isfinite(read_value)
        and math.isfinite(write_value)
    ):
        return None

    if read_value < 0 or write_value < 0:
        return None

    total = read_value + write_value

    if total <= 0:
        return None

    # Explicit percentages already summing to 100 are preserved.
    # Other positive ratios such as 4:1 are normalized to percentages.
    read_percent = (
        read_value
        if math.isclose(total, 100.0, abs_tol=1e-6)
        else (read_value / total) * 100.0
    )
    write_percent = (
        write_value
        if math.isclose(total, 100.0, abs_tol=1e-6)
        else (write_value / total) * 100.0
    )

    return {
        "read_percent": round(read_percent, 6),
        "write_percent": round(write_percent, 6),
    }


def parse_read_write_ratio(
    text: str,
    *,
    pending_ratio_question: bool = False,
) -> Optional[dict]:
    """
    Parse an explicit read/write pair into canonical percentages.

    Supported examples:
      20/80
      80:20
      read 20 write 80
      20 read 80 write
      lecture 20 écriture 80
      write 80 read 20

    A bare ``20/80`` form is accepted only when the active clarification
    explicitly targets read_write_ratio. This avoids treating arbitrary
    fractions in unrelated messages as an I/O ratio.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    match = _READ_THEN_WRITE.search(raw)
    if match:
        return _normalize_pair(
            float(match.group(1)),
            float(match.group(2)),
        )

    match = _NUM_READ_NUM_WRITE.search(raw)
    if match:
        return _normalize_pair(
            float(match.group(1)),
            float(match.group(2)),
        )

    match = _WRITE_THEN_READ.search(raw)
    if match:
        return _normalize_pair(
            float(match.group(2)),
            float(match.group(1)),
        )

    match = _NUM_WRITE_NUM_READ.search(raw)
    if match:
        return _normalize_pair(
            float(match.group(2)),
            float(match.group(1)),
        )

    if pending_ratio_question:
        match = _RATIO_SEP.fullmatch(raw)
        if match:
            return _normalize_pair(
                float(match.group(1)),
                float(match.group(2)),
            )

    # Outside a pending question, require explicit read/write semantics.
    lowered = raw.casefold()
    has_read = "read" in lowered or "lecture" in lowered
    has_write = (
        "write" in lowered
        or "ecriture" in lowered
        or "écriture" in lowered
    )

    if has_read and has_write:
        numbers = re.findall(
            r"\d+(?:\.\d+)?",
            raw,
        )
        if len(numbers) == 2:
            return _normalize_pair(
                float(numbers[0]),
                float(numbers[1]),
            )

    return None
