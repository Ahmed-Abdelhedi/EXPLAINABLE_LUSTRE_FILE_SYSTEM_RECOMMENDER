from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from full_architecture.catalog_loader import (  # noqa: E402
    DEFAULT_HARDWARE_CATALOG,
    DEFAULT_HARDWARE_MANIFEST,
    load_reference_catalog,
    sha256_file,
)


def test_reference_catalog_loads() -> None:
    catalog = load_reference_catalog()
    assert catalog["catalog_version"] == "1.0.1"


def test_reference_catalog_provenance_is_explicit() -> None:
    catalog = load_reference_catalog()
    assert (
        catalog["provenance"]["status"]
        == "POLICY_REFERENCE_NOT_VENDOR_BENCHMARKED"
    )


def test_reference_catalog_component_ids_are_globally_unique() -> None:
    catalog = load_reference_catalog()
    ids: list[str] = []

    for section in (
        "servers",
        "controllers",
        "enclosures",
        "networks",
        "protection_profiles",
        "ha_profiles",
    ):
        ids.extend(
            item["id"]
            for item in catalog[
                section
            ]
        )

    assert len(ids) == len(
        set(
            ids
        )
    )


def test_reference_catalog_covers_mds_and_oss() -> None:
    catalog = load_reference_catalog()

    roles = {
        item["role"]
        for item in catalog[
            "servers"
        ]
    }

    assert "MDS" in roles
    assert "OSS" in roles


def test_reference_catalog_covers_all_drive_protocols() -> None:
    catalog = load_reference_catalog()

    protocols: set[str] = set()

    for item in catalog[
        "controllers"
    ]:
        protocols.update(
            item[
                "supported_protocols"
            ]
        )

    assert {
        "SATA",
        "SAS",
        "NVME",
    }.issubset(
        protocols
    )


def test_reference_catalog_has_ethernet_and_infiniband() -> None:
    catalog = load_reference_catalog()

    fabrics = {
        item["fabric"]
        for item in catalog[
            "networks"
        ]
    }

    assert "ETHERNET" in fabrics
    assert "INFINIBAND" in fabrics


def test_reference_catalog_has_core_protection_profiles() -> None:
    catalog = load_reference_catalog()

    levels = {
        item["raid_level"]
        for item in catalog[
            "protection_profiles"
        ]
    }

    assert {
        "RAID1",
        "RAID10",
        "RAID6",
    }.issubset(
        levels
    )


def test_reference_catalog_has_gen5_nvme_controller() -> None:
    catalog = load_reference_catalog()

    controllers = [
        item
        for item in catalog[
            "controllers"
        ]
        if (
            "NVME"
            in item[
                "supported_protocols"
            ]
        )
    ]

    assert any(
        int(
            item[
                "pcie_gen"
            ]
        )
        >= 5
        for item in controllers
    )


def test_reference_catalog_has_e1s_e3s_nvme_enclosure() -> None:
    catalog = load_reference_catalog()

    enclosures = [
        item
        for item in catalog[
            "enclosures"
        ]
        if (
            "NVME"
            in item[
                "supported_protocols"
            ]
        )
    ]

    supported = set()

    for item in enclosures:
        supported.update(
            item[
                "supported_form_factors"
            ]
        )

    assert "FF_E1S" in supported
    assert "FF_E3S" in supported


def test_manifest_sha256_matches_catalog() -> None:
    with DEFAULT_HARDWARE_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    assert (
        sha256_file(
            DEFAULT_HARDWARE_CATALOG
        )
        == manifest[
            "catalog_sha256"
        ]
    )
