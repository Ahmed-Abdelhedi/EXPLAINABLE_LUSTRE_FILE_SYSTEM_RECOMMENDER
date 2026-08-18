from requirement_extractor_v2.deterministic_verifier import (
    DeterministicVerifier,
)

from requirement_extractor_v2.models import (
    ParamName,
    Quantity,
    QuantityDimension,
    SemanticLink,
    SemanticRole,
)


def show(
    verifier,
    title,
    quantity,
    link,
    text,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    decision = verifier.verify(
        quantity=quantity,
        link=link,
        source_text=text,
    )

    print(decision.to_dict())


def main():

    verifier = DeterministicVerifier()

    # ================================================================
    # CASE 1 — VALID POWER
    # ================================================================

    text = "Maximum power is 1.5 kW."

    q1 = Quantity(
        id="q1",
        raw="1.5 kW",
        value=1.5,
        unit="kW",
        dimension=QuantityDimension.POWER,
        start=17,
        end=23,
        source_text=text,
    )

    link1 = SemanticLink(
        quantity_id="q1",
        field=ParamName.max_power_w,
        role=SemanticRole.MAXIMUM_LIMIT,
        evidence=text,
        resolver="explicit_pattern",
    )

    show(
        verifier,
        "CASE 1 — VALID POWER",
        q1,
        link1,
        text,
    )

    # ================================================================
    # CASE 2 — VALID CLIENT COUNT
    # ================================================================

    text = "Around 275 endpoints will mount the filesystem."

    q2 = Quantity(
        id="q2",
        raw="275",
        value=275,
        unit=None,
        dimension=QuantityDimension.UNKNOWN,
        start=7,
        end=10,
        source_text=text,
    )

    link2 = SemanticLink(
        quantity_id="q2",
        field=ParamName.client_count,
        role=SemanticRole.TOTAL_COUNT,
        evidence=text,
        resolver="semantic_linker_xlmr",
    )

    show(
        verifier,
        "CASE 2 — VALID CLIENT COUNT",
        q2,
        link2,
        text,
    )

    # ================================================================
    # CASE 3 — INVALID DECIMAL COUNT
    # ================================================================

    text = "We expect 12.5 clients."

    q3 = Quantity(
        id="q3",
        raw="12.5",
        value=12.5,
        unit=None,
        dimension=QuantityDimension.UNKNOWN,
        start=10,
        end=14,
        source_text=text,
    )

    link3 = SemanticLink(
        quantity_id="q3",
        field=ParamName.client_count,
        role=SemanticRole.TOTAL_COUNT,
        evidence=text,
        resolver="semantic_linker_xlmr",
    )

    show(
        verifier,
        "CASE 3 — INVALID DECIMAL COUNT",
        q3,
        link3,
        text,
    )

    # ================================================================
    # CASE 4 — FIELD/DIMENSION MISMATCH
    # ================================================================

    text = "Power limit is 1500 W."

    q4 = Quantity(
        id="q4",
        raw="1500 W",
        value=1500,
        unit="W",
        dimension=QuantityDimension.POWER,
        start=15,
        end=21,
        source_text=text,
    )

    link4 = SemanticLink(
        quantity_id="q4",
        field=ParamName.max_budget_usd,
        role=SemanticRole.MAXIMUM_LIMIT,
        evidence=text,
        resolver="test",
    )

    show(
        verifier,
        "CASE 4 — DIMENSION MISMATCH",
        q4,
        link4,
        text,
    )

    # ================================================================
    # CASE 5 — UNSUPPORTED EVIDENCE
    # ================================================================

    text = "Around 300 clients will connect."

    q5 = Quantity(
        id="q5",
        raw="300",
        value=300,
        unit=None,
        dimension=QuantityDimension.UNKNOWN,
        start=7,
        end=10,
        source_text=text,
    )

    link5 = SemanticLink(
        quantity_id="q5",
        field=ParamName.client_count,
        role=SemanticRole.TOTAL_COUNT,
        evidence="500 clients",
        resolver="llm_fallback",
    )

    show(
        verifier,
        "CASE 5 — BAD EVIDENCE",
        q5,
        link5,
        text,
    )


if __name__ == "__main__":
    main()