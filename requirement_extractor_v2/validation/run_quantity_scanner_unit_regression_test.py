from __future__ import annotations

from requirement_extractor_v2.models import QuantityDimension
from requirement_extractor_v2.quantity_scanner import QuantityScanner


def assert_quantity(
    quantity,
    *,
    value,
    unit,
    dimension,
):
    assert quantity.value == value, (
        f"value: expected={value!r}, got={quantity.value!r}"
    )
    assert quantity.unit == unit, (
        f"unit: expected={unit!r}, got={quantity.unit!r}"
    )
    assert quantity.dimension == dimension, (
        f"dimension: expected={dimension!r}, got={quantity.dimension!r}"
    )


def main():
    scanner = QuantityScanner()

    # 1. Fuzzy number + symbolic capacity unit.
    quantities = scanner.scan(
        "The usable target is five hunderd TiB."
    )
    assert len(quantities) == 1
    assert_quantity(
        quantities[0],
        value=500,
        unit="TiB",
        dimension=QuantityDimension.CAPACITY,
    )
    assert quantities[0].corrected is True
    print("[PASS] fuzzy number + TiB")

    # 2. Fuzzy number + symbolic file-size unit.
    quantities = scanner.scan(
        "The maximum file size is eight hunderd GB."
    )
    assert len(quantities) == 1
    assert_quantity(
        quantities[0],
        value=800,
        unit="GB",
        dimension=QuantityDimension.FILE_SIZE,
    )
    assert quantities[0].corrected is True
    print("[PASS] fuzzy number + GB")

    # 3. Shared throughput unit.
    quantities = scanner.scan(
        "Read throughput should be 20 or 30 GB/s."
    )
    assert len(quantities) == 2
    assert_quantity(
        quantities[0],
        value=20,
        unit="GB/s",
        dimension=QuantityDimension.THROUGHPUT,
    )
    assert_quantity(
        quantities[1],
        value=30,
        unit="GB/s",
        dimension=QuantityDimension.THROUGHPUT,
    )
    print("[PASS] shared GB/s alternative")

    # 4. Shared file-size unit.
    quantities = scanner.scan(
        "Maximum file size could be 50 or 100 GB."
    )
    assert len(quantities) == 2
    assert_quantity(
        quantities[0],
        value=50,
        unit="GB",
        dimension=QuantityDimension.FILE_SIZE,
    )
    assert_quantity(
        quantities[1],
        value=100,
        unit="GB",
        dimension=QuantityDimension.FILE_SIZE,
    )
    print("[PASS] shared GB alternative")

    # 5. French shared throughput unit.
    quantities = scanner.scan(
        "Le débit d'écriture sera de 40 ou 60 GB/s."
    )
    assert len(quantities) == 2
    assert_quantity(
        quantities[0],
        value=40,
        unit="GB/s",
        dimension=QuantityDimension.THROUGHPUT,
    )
    assert_quantity(
        quantities[1],
        value=60,
        unit="GB/s",
        dimension=QuantityDimension.THROUGHPUT,
    )
    print("[PASS] shared French GB/s alternative")

    # 6. French shared power unit.
    quantities = scanner.scan(
        "La puissance maximale sera 800 ou 1200 W."
    )
    assert len(quantities) == 2
    assert_quantity(
        quantities[0],
        value=800,
        unit="W",
        dimension=QuantityDimension.POWER,
    )
    assert_quantity(
        quantities[1],
        value=1200,
        unit="W",
        dimension=QuantityDimension.POWER,
    )
    print("[PASS] shared French W alternative")

    # 7. Safety: unrelated quantities must not share units.
    quantities = scanner.scan(
        "We need 200 clients and maximum power is 800 W."
    )
    assert len(quantities) == 2
    assert_quantity(
        quantities[0],
        value=200,
        unit=None,
        dimension=QuantityDimension.UNKNOWN,
    )
    assert_quantity(
        quantities[1],
        value=800,
        unit="W",
        dimension=QuantityDimension.POWER,
    )
    print("[PASS] no unsafe unit propagation")

    # 8. Safety: a bare contextual value stays unitless in the scanner.
    quantities = scanner.scan("200")
    assert len(quantities) == 1
    assert_quantity(
        quantities[0],
        value=200,
        unit=None,
        dimension=QuantityDimension.UNKNOWN,
    )
    print("[PASS] bare contextual quantity remains unitless")

    # 9. Count alternatives remain unitless.
    quantities = scanner.scan(
        "We may have 200 or 300 clients."
    )
    assert len(quantities) == 2
    assert quantities[0].unit is None
    assert quantities[1].unit is None
    print("[PASS] count alternatives remain unitless")

    print()
    print("QuantityScanner unit regression: ALL TESTS PASSED")


if __name__ == "__main__":
    main()