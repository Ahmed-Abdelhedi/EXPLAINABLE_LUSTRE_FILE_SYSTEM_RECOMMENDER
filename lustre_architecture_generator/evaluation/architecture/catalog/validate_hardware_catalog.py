from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.catalog_loader import (  # noqa: E402
    DEFAULT_HARDWARE_CATALOG,
    DEFAULT_HARDWARE_MANIFEST,
    load_reference_catalog,
    sha256_file,
)


def main() -> None:
    catalog = load_reference_catalog()

    print("FULL ARCHITECTURE HARDWARE CATALOG H4")
    print("=====================================")
    print("Status              : VALIDATED")
    print("Catalog version     :", catalog["catalog_version"])
    print("Catalog kind        :", catalog["catalog_kind"])
    print(
        "Provenance          :",
        catalog["provenance"]["status"],
    )
    print("Servers             :", len(catalog["servers"]))
    print(
        "Controllers         :",
        len(catalog["controllers"]),
    )
    print(
        "Enclosures          :",
        len(catalog["enclosures"]),
    )
    print("Networks            :", len(catalog["networks"]))
    print(
        "Protection profiles :",
        len(catalog["protection_profiles"]),
    )
    print(
        "HA profiles         :",
        len(catalog["ha_profiles"]),
    )
    print(
        "Catalog SHA256      :",
        sha256_file(DEFAULT_HARDWARE_CATALOG),
    )
    print("Manifest            :", DEFAULT_HARDWARE_MANIFEST)


if __name__ == "__main__":
    main()
