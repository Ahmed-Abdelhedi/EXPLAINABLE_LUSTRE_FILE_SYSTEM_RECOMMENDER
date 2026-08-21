from __future__ import annotations

from requirement_extractor_v2.explicit_pattern_resolver import (
    ExplicitPatternResolver,
)
from requirement_extractor_v2.models import (
    Quantity,
    QuantityDetection,
    QuantityDimension,
)


def build_quantity(
    text: str,
    raw: str,
    value: float,
) -> Quantity:
    start = text.index(
        raw
    )

    return Quantity(
        id="q1",
        raw=raw,
        normalized=raw,
        value=value,
        unit="GB",
        dimension=(
            QuantityDimension.FILE_SIZE
        ),
        start=start,
        end=start + len(raw),
        source_text=text,
        detection=(
            QuantityDetection.UNKNOWN
        ),
        corrected=False,
    )


CASES = [
    (
        "average singular du fichier",
        "La taille moyenne du fichier est 1.5 GB.",
        "1.5 GB",
        1.5,
        "average_file_size_gb",
        "average_value",
    ),
    (
        "maximum singular du fichier",
        "La taille maximum du fichier est 75 GB.",
        "75 GB",
        75,
        "max_file_size_gb",
        "maximum_limit",
    ),
    (
        "maximale singular du fichier",
        "La taille maximale du fichier est 80 GB.",
        "80 GB",
        80,
        "max_file_size_gb",
        "maximum_limit",
    ),
    (
        "average de la fichier phrase variant",
        "La taille moyenne de la fichier test est 2 GB.",
        "2 GB",
        2,
        None,
        None,
    ),
    (
        "ambiguous file size stays unresolved",
        "Une taille de fichier de 50 GB est mentionnée.",
        "50 GB",
        50,
        None,
        None,
    ),
]

resolver = (
    ExplicitPatternResolver()
)

failures = []

for (
    name,
    text,
    raw,
    value,
    expected_field,
    expected_role,
) in CASES:

    quantity = build_quantity(
        text,
        raw,
        value,
    )

    result = resolver.resolve(
        text,
        [quantity],
    )

    link = (
        result.links[0]
        if result.links
        else None
    )

    if expected_field is None:
        passed = (
            link is None
        )
    else:
        passed = (
            link is not None
            and
            link.field.value
            == expected_field
            and
            link.role.value
            == expected_role
        )

    print(
        "[PASS]" if passed else "[FAIL]",
        name,
        "->",
        (
            "UNRESOLVED"
            if link is None
            else (
                f"{link.field.value} / "
                f"{link.role.value}"
            )
        ),
    )

    if not passed:
        failures.append(
            name
        )

if failures:
    raise SystemExit(
        "Failures: "
        + ", ".join(
            failures
        )
    )

print()
print(
    "EXPLICIT RESOLVER FRENCH REGRESSION: "
    "ALL TESTS PASSED"
)