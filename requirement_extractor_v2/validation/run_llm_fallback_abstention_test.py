from requirement_extractor_v2.quantity_scanner import QuantityScanner
from requirement_extractor_v2.llm_fallback_extractor import (
    LLMFallbackExtractor,
)


def main():
    print("=" * 70)
    print("LLM FALLBACK — ABSTENTION TEST")
    print("=" * 70)

    text = "The requirement is around 320."

    scanner = QuantityScanner()
    quantities = scanner.scan(text)

    print("\n[1] QUANTITY SCANNER")

    for quantity in quantities:
        print(quantity.to_dict())

    if not quantities:
        raise RuntimeError("No quantity detected.")

    quantity = quantities[0]

    fallback = LLMFallbackExtractor()

    print("\n[2] LLM FALLBACK RESULT")

    link = fallback.resolve_quantity(
        user_text=text,
        quantity=quantity,
    )

    if link is None:
        print("✅ UNRESOLVED — correct abstention")
    else:
        print("❌ SHOULD HAVE ABSTAINED")
        print(link.to_dict())

    print("\n[3] CALL LOG")

    for entry in fallback.call_log:
        print(entry)


if __name__ == "__main__":
    main()