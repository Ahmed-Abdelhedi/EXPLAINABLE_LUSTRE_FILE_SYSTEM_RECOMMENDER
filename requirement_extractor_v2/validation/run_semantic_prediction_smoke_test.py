from requirement_extractor_v2.quantity_scanner import (
    QuantityScanner,
)

from requirement_extractor_v2.explicit_pattern_resolver import (
    ExplicitPatternResolver,
)

from requirement_extractor_v2.semantic_linker.runtime import (
    SemanticLinkerRuntime,
)


def main():
    print("=" * 70)
    print("SEMANTIC LINKER — REAL PREDICTION TEST")
    print("=" * 70)

    text = (
        "Around 275 endpoints will mount "
        "the shared filesystem."
    )

    # --------------------------------------------------
    # 1. Quantity Scanner
    # --------------------------------------------------

    scanner = QuantityScanner()

    quantities = scanner.scan(text)

    print("\n[1] QUANTITY SCANNER")

    for quantity in quantities:
        print(quantity.to_dict())

    if not quantities:
        raise RuntimeError(
            "No quantity detected."
        )

    # --------------------------------------------------
    # 2. Explicit Resolver
    # --------------------------------------------------

    explicit_resolver = (
        ExplicitPatternResolver()
    )

    explicit_result = (
        explicit_resolver.resolve(
            text,
            quantities,
        )
    )

    print("\n[2] EXPLICIT RESOLVER")
    print(explicit_result.to_dict())

    # --------------------------------------------------
    # 3. Semantic Linker
    # --------------------------------------------------

    linker = SemanticLinkerRuntime()

    print("\n[3] SEMANTIC LINKER")

    for quantity in quantities:

        if (
            quantity.id
            not in
            explicit_result.unresolved_quantity_ids
        ):
            print(
                f"{quantity.id}: already resolved "
                "by explicit resolver"
            )
            continue

        prediction = linker.predict(
            text=text,
            quantity=quantity,
        )

        print(
            prediction.to_dict()
        )


if __name__ == "__main__":
    main()