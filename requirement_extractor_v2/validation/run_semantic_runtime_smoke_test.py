from requirement_extractor_v2.semantic_linker.runtime import (
    SemanticLinkerRuntime,
)


def main():
    print("=" * 70)
    print("SEMANTIC LINKER RUNTIME — LOAD TEST")
    print("=" * 70)

    linker = SemanticLinkerRuntime()

    print("\n✅ Model loaded successfully\n")

    for key, value in linker.info().items():
        print(f"{key:24s}: {value}")


if __name__ == "__main__":
    main()