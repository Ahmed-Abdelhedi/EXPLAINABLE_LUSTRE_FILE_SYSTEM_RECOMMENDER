from requirement_extractor_v2.selective_cascade import (
    SelectiveCascade,
)


def show_case(
    cascade,
    title,
    text,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(text)

    before_llm = (
        cascade
        .llm_fallback
        .call_count
    )

    result = cascade.resolve(text)

    after_llm = (
        cascade
        .llm_fallback
        .call_count
    )

    print("\nLINKS")

    for link in result.links:
        print(link.to_dict())

    print(
        "\nUNRESOLVED:",
        result.unresolved_quantity_ids,
    )

    print("\nTRACES")

    for trace in result.traces.values():
        print(trace.to_dict())

    print(
        "\nLLM calls for this case:",
        after_llm - before_llm,
    )


def main():

    cascade = SelectiveCascade()

    # --------------------------------------------------
    # CASE 1
    # Explicit resolver must stop the cascade.
    # --------------------------------------------------

    show_case(
        cascade,
        "CASE 1 — EXPLICIT",
        "Maximum power is 1500 W.",
    )

    # --------------------------------------------------
    # CASE 2
    # Explicit abstains, XLM-R should resolve.
    # LLM MUST NOT run.
    # --------------------------------------------------

    show_case(
        cascade,
        "CASE 2 — SEMANTIC LINKER",
        (
            "Around 275 endpoints will mount "
            "the shared filesystem."
        ),
    )

    # --------------------------------------------------
    # CASE 3
    # Explicit abstains.
    # Semantic should abstain/reject.
    # LLM should run and also abstain.
    # --------------------------------------------------

    show_case(
        cascade,
        "CASE 3 — FINAL UNRESOLVED",
        "The requirement is around 320.",
    )


if __name__ == "__main__":
    main()