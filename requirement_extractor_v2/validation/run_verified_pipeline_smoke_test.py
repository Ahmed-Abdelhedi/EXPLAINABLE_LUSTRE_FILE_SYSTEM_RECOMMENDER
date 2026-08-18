from requirement_extractor_v2.verified_pipeline import (
    VerifiedRequirementPipeline,
)


def run_case(
    pipeline,
    title,
    text,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(text)

    before_llm = (
        pipeline
        .cascade
        .llm_fallback
        .call_count
    )

    result = pipeline.process(text)

    after_llm = (
        pipeline
        .cascade
        .llm_fallback
        .call_count
    )

    print("\nDECISIONS")

    for decision in result.decisions:
        print(decision.to_dict())

    print(
        "\nVERIFIED VALUES:",
        result.verified_values(),
    )

    print(
        "\nSUMMARY:",
        result.to_dict()["summary"],
    )

    print(
        "\nLLM calls:",
        after_llm - before_llm,
    )


def main():

    pipeline = VerifiedRequirementPipeline()

    # ================================================================
    # CASE 1
    # Explicit → verifier
    # ================================================================

    run_case(
        pipeline,
        "CASE 1 — EXPLICIT + VERIFIED",
        "Maximum power is 1.5 kW.",
    )

    # ================================================================
    # CASE 2
    # XLM-R → verifier
    # ================================================================

    run_case(
        pipeline,
        "CASE 2 — SEMANTIC + VERIFIED",
        (
            "Around 275 endpoints will mount "
            "the shared filesystem."
        ),
    )

    # ================================================================
    # CASE 3
    # XLM-R reject → LLM abstain → unresolved
    # ================================================================

    run_case(
        pipeline,
        "CASE 3 — FINAL UNRESOLVED",
        "The requirement is around 320.",
    )


if __name__ == "__main__":
    main()