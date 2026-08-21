from __future__ import annotations

from requirement_extractor_v2.deterministic_verifier import DeterministicVerifier
from requirement_extractor_v2.models import (
    ParamName,
    Quantity,
    QuantityDimension,
    SemanticLink,
    SemanticRole,
    VerificationStatus,
)


def check(
    name,
    text,
    raw,
    value,
    unit,
    dimension,
    field,
    role,
    expected,
):
    start = max(text.find(raw), 0)

    quantity = Quantity(
        id="q1",
        raw=raw,
        value=value,
        unit=unit,
        dimension=QuantityDimension(dimension),
        start=start,
        end=start + len(raw),
        source_text=text,
    )

    link = SemanticLink(
        quantity_id="q1",
        field=ParamName(field),
        role=SemanticRole(role),
        evidence=text,
        resolver="unit_guard_regression",
    )

    decision = DeterministicVerifier().verify(
        quantity=quantity,
        link=link,
        source_text=text,
    )

    assert decision.status == expected, (
        f"{name}: expected={expected.value}, "
        f"got={decision.status.value}, reasons={decision.reasons}"
    )

    print(f"[PASS] {name:<34} -> {decision.status.value}")


def main():
    invalid = [
        ("power cannot accept GB", "Maximum power is 800 GB.", "800 GB", 800, "GB", "power", "max_power_w", "maximum_limit"),
        ("throughput cannot accept W", "Read throughput is 70 W.", "70 W", 70, "W", "throughput", "target_read_gbps", "target"),
        ("capacity cannot accept USD", "Usable capacity is 500 USD.", "500 USD", 500, "USD", "capacity", "requested_usable_capacity_tib", "target"),
        ("file size cannot accept W", "Maximum file size is 50 W.", "50 W", 50, "W", "file_size", "max_file_size_gb", "maximum_limit"),
        ("budget cannot accept GB", "Budget is 100000 GB.", "100000 GB", 100000, "GB", "money", "max_budget_usd", "maximum_limit"),
        ("growth cannot accept W", "Annual growth is 30 W.", "30 W", 30, "W", "percent", "annual_growth_percent", "growth_rate"),
        ("ratio cannot accept W", "Read ratio is 70 W.", "70 W", 70, "W", "percent", "read_write_ratio", "ratio_component"),
        ("count cannot accept W", "Client count is 200 W.", "200 W", 200, "W", "unknown", "client_count", "total_count"),
    ]

    for args in invalid:
        check(*args, VerificationStatus.INVALID)

    valid = [
        ("valid kW normalization", "Maximum power is 1.5 kW.", "1.5 kW", 1.5, "kW", "power", "max_power_w", "maximum_limit"),
        ("valid TiB", "Usable capacity is 500 TiB.", "500 TiB", 500, "TiB", "capacity", "requested_usable_capacity_tib", "target"),
        ("valid GB/s", "Read throughput is 70 GB/s.", "70 GB/s", 70, "GB/s", "throughput", "target_read_gbps", "target"),
        ("valid percent", "Annual growth is 30 percent.", "30 percent", 30, "%", "percent", "annual_growth_percent", "growth_rate"),
        ("valid unitless count", "Client count is 200.", "200", 200, None, "unknown", "client_count", "total_count"),
    ]

    for args in valid:
        check(*args, VerificationStatus.VERIFIED)

    print()
    print("VERIFIER SOURCE-UNIT REGRESSION: ALL TESTS PASSED")


if __name__ == "__main__":
    main()