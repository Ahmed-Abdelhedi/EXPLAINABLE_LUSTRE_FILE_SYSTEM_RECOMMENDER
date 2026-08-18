from requirement_extractor_v2.quantity_scanner import QuantityScanner
from requirement_extractor_v2.llm_fallback_extractor import (
    LLMFallbackExtractor,
)


def main():
    print("=" * 70)
    print("LLM FALLBACK — REAL OLLAMA TEST")
    print("=" * 70)

    text = (
        "Approximately 320 compute clients "
        "will access the shared filesystem."
    )

    # --------------------------------------------------
    # 1. Detect quantity
    # --------------------------------------------------

    scanner = QuantityScanner()
    quantities = scanner.scan(text)

    print("\n[1] QUANTITY SCANNER")

    for quantity in quantities:
        print(quantity.to_dict())

    if not quantities:
        raise RuntimeError("No quantity detected.")

    quantity = quantities[0]

    # --------------------------------------------------
    # 2. LLM fallback
    # --------------------------------------------------

    fallback = LLMFallbackExtractor()

    print("\n[2] LLM FALLBACK CONFIG")
    print(fallback.info())

    print("\n[3] LLM FALLBACK RESULT")

    link = fallback.resolve_quantity(
        user_text=text,
        quantity=quantity,
    )

    if link is None:
        print("UNRESOLVED")
    else:
        print(link.to_dict())

    # --------------------------------------------------
    # 3. Debug
    # --------------------------------------------------

    print("\n[4] CALL INFORMATION")
    print(f"call_count = {fallback.call_count}")

    print("\n[5] CALL LOG")

    for entry in fallback.call_log:
        print(entry)


if __name__ == "__main__":
    main()