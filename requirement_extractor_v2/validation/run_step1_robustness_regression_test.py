from __future__ import annotations

from requirement_extractor_v2.explicit_pattern_resolver import ExplicitPatternResolver
from requirement_extractor_v2.models import ParamName, QuantityDimension
from requirement_extractor_v2.quantity_scanner import QuantityScanner
from requirement_extractor_v2.robust_explicit_pattern_resolver import (
    RobustExplicitPatternResolver,
)
from requirement_extractor_v2.robust_quantity_scanner import RobustQuantityScanner


def _assert_quantity(scanner, text, value, unit, dimension):
    quantities = scanner.scan(text)
    assert len(quantities) == 1, (text, [q.to_dict() for q in quantities])
    quantity = quantities[0]
    assert quantity.value == value, quantity.to_dict()
    assert quantity.unit == unit, quantity.to_dict()
    assert quantity.dimension == dimension, quantity.to_dict()
    assert quantity.corrected is True, quantity.to_dict()
    return quantity


def run() -> None:
    scanner = RobustQuantityScanner(QuantityScanner())

    _assert_quantity(
        scanner,
        "Maximum power is 800 wats.",
        800,
        "W",
        QuantityDimension.POWER,
    )
    _assert_quantity(
        scanner,
        "Maximum file size is fourty GB.",
        40,
        "GB",
        QuantityDimension.FILE_SIZE,
    )
    _assert_quantity(
        scanner,
        "Write throughput should be fourty GB/s.",
        40,
        "GB/s",
        QuantityDimension.THROUGHPUT,
    )

    resolver = RobustExplicitPatternResolver(ExplicitPatternResolver())

    text = "Read target: seventy GB/s et write target: thirty five GB/s."
    quantities = scanner.scan(text)
    result = resolver.resolve(text, quantities)
    mapping = {link.quantity_id: link.field for link in result.links}
    assert len(quantities) == 2, [q.to_dict() for q in quantities]
    assert mapping[quantities[0].id] == ParamName.target_read_gbps, [
        link.to_dict() for link in result.links
    ]
    assert mapping[quantities[1].id] == ParamName.target_write_gbps, [
        link.to_dict() for link in result.links
    ]

    text = "Taille moyenne 3 GB et taille maximale 90 GB."
    quantities = scanner.scan(text)
    result = resolver.resolve(text, quantities)
    mapping = {link.quantity_id: link.field for link in result.links}
    assert len(quantities) == 2, [q.to_dict() for q in quantities]
    assert mapping[quantities[0].id] == ParamName.average_file_size_gb, [
        link.to_dict() for link in result.links
    ]
    assert mapping[quantities[1].id] == ParamName.max_file_size_gb, [
        link.to_dict() for link in result.links
    ]

    # Safety regressions: ordinary words near numbers must not become numbers.
    assert scanner.scan("This is for GB storage.") == []
    assert scanner.scan("The file is GB formatted.") == []

    print("Step1 robustness regression test: PASS")
    print("  - digit fuzzy unit: PASS")
    print("  - fuzzy written number + explicit unit: PASS")
    print("  - read/write local structural binding: PASS")
    print("  - average/max file-size local binding: PASS")
    print("  - anti-false-positive blocklist: PASS")


if __name__ == "__main__":
    run()
