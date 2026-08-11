import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent


USECASE_FILE = (
    BASE_DIR
    /
    "data"
    /
    "use_cases_lustre_1200_v4_preferences.json"
)


CATALOG_FILE = (
    BASE_DIR
    /
    "data"
    /
    "catalogue_drives_model_ready_clean.json"
)


OUTPUT_FILE = (
    BASE_DIR
    /
    "output"
    /
    "drive_selector_dataset_v4.csv"
)


# =====================================================
# Global configuration
# =====================================================

DEFAULT_RAID_EFFICIENCY = 0.75



# =====================================================
# Load data
# =====================================================

with open(USECASE_FILE, encoding="utf-8") as f:
    use_cases = json.load(f)


with open(CATALOG_FILE, encoding="utf-8") as f:
    drives = json.load(f)


print(
    f"Use cases loaded : {len(use_cases)}"
)

print(
    f"Drives loaded : {len(drives)}"
)



# =====================================================
# Helper functions
# =====================================================


def safe_get(data, key, default=0):
    """
    Avoid missing values in catalogue
    """
    value = data.get(key, default)

    if value is None:
        return default

    return value



# =====================================================
# Deterministic filtering
# =====================================================


def drive_feasible(drive, case):


    capacity_tb = (
        safe_get(
            drive,
            "capacity_en_GB"
        )
        /
        1024
    )


    if capacity_tb <= 0:
        return None



    price = safe_get(
        drive,
        "price_en_dollars"
    )


    power = safe_get(
        drive,
        "power_consumption_en_w"
    )



    # ---------------------------------------------
    # Capacity with future growth
    # ---------------------------------------------

    growth_factor = (
        1
        +
        case["annual_growth_percent"] / 100
    )


    required_capacity = (
        case["requested_usable_capacity_tib"]
        *
        growth_factor
    )



    # ---------------------------------------------
    # Number of drives required
    # ---------------------------------------------

    required_drives = (
        required_capacity
        /
        (
            capacity_tb
            *
            DEFAULT_RAID_EFFICIENCY
        )
    )


    required_drives = int(
        required_drives + 0.999
    )



    estimated_cost = (
        required_drives
        *
        price
    )


    estimated_power = (
        required_drives
        *
        power
    )



    # ---------------------------------------------
    # Hard constraints
    # ---------------------------------------------

    if estimated_cost > case["max_budget_usd"]:
        return None


    if estimated_power > case["max_power_w"]:
        return None



    return {

        "estimated_drive_count":
            required_drives,

        "estimated_cost":
            round(
                estimated_cost,
                2
            ),

        "estimated_power":
            round(
                estimated_power,
                2
            ),

        "required_capacity_with_growth":
            round(
                required_capacity,
                2
            )
    }




# =====================================================
# Feature extraction
# =====================================================


def build_row(case, drive, filter_info):


    return {


        # ==============================
        # Case ID
        # ==============================

        "case_id":
            case["case_id"],



        # ==============================
        # User requirements
        # ==============================

        "requested_capacity_tib":
            case["requested_usable_capacity_tib"],


        "client_count":
            case["client_count"],


        "average_file_size_gb":
            case["average_file_size_gb"],


        "max_file_size_gb":
            case["max_file_size_gb"],


        "total_file_count":
            case["total_file_count"],


        "read_percent":
            case["read_write_ratio"]["read_percent"],


        "write_percent":
            case["read_write_ratio"]["write_percent"],


        "access_type":
            case["access_type"],


        "target_read_gbps":
            case["target_read_gbps"],


        "target_write_gbps":
            case["target_write_gbps"],


        "ha_required":
            int(case["ha_required"]),


        "budget":
            case["max_budget_usd"],


        "power_limit":
            case["max_power_w"],


        "annual_growth_percent":
            case["annual_growth_percent"],




        # ==============================
        # User preferences
        # ==============================


        "performance_priority":
            case["performance_priority"],


        "cost_priority":
            case["cost_priority"],


        "power_priority":
            case["power_priority"],


        "reliability_priority":
            case["reliability_priority"],




        # ==============================
        # Drive features
        # ==============================


        "drive_name":
            drive["name"],


        "manufacturer":
            drive["manufacturer"],


        "media_type":
            drive["media_type"],


        "protocol":
            drive["protocol"],


        "capacity_tb":
            safe_get(
                drive,
                "capacity_en_GB"
            )
            /
            1024,


        "seq_read":
            safe_get(
                drive,
                "seq_read_mb_s"
            ),


        "seq_write":
            safe_get(
                drive,
                "seq_write_mb_s"
            ),


        "random_read_iops":
            safe_get(
                drive,
                "random_read_iops_4k"
            ),


        "random_write_iops":
            safe_get(
                drive,
                "random_write_iops_4k"
            ),


            "price":
                safe_get(
                    drive,
                    "price_en_dollars"
                ),


            "power":
                safe_get(
                    drive,
                    "power_consumption_en_w"
                ),


        "life_expectancy":
            safe_get(
                drive,
                "life_expectancy"
            ),




        # ==============================
        # Filtering explanation
        # ==============================


        "estimated_drive_count":
            filter_info["estimated_drive_count"],


        "estimated_cost":
            filter_info["estimated_cost"],


        "estimated_power":
            filter_info["estimated_power"],


        "required_capacity_with_growth":
            filter_info["required_capacity_with_growth"]

    }



# =====================================================
# Dataset generation
# =====================================================


rows = []


no_candidate_cases = []


for case in tqdm(use_cases):


    candidate_count = 0


    for drive in drives:


        result = drive_feasible(
            drive,
            case
        )


        if result is not None:


            rows.append(
                build_row(
                    case,
                    drive,
                    result
                )
            )


            candidate_count += 1



    if candidate_count == 0:

        no_candidate_cases.append(
            case["case_id"]
        )



# =====================================================
# Save
# =====================================================


dataset = pd.DataFrame(rows)



print()
print(
    "Final rows:",
    len(dataset)
)


print(
    "Columns:",
    len(dataset.columns)
)


print(
    "Cases without candidate:",
    len(no_candidate_cases)
)



dataset.to_csv(
    OUTPUT_FILE,
    index=False
)



print(
    "Saved:",
    OUTPUT_FILE
)