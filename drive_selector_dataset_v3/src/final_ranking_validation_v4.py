import pandas as pd
from pathlib import Path


# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent


INPUT_FILE = (
    BASE_DIR /
    "output" /
    "drive_selector_dataset_v4_ranked.csv"
)


REPORT_FILE = (
    BASE_DIR /
    "output" /
    "final_ranking_validation_v4.txt"
)


# =====================================================
# Load
# =====================================================

df = pd.read_csv(INPUT_FILE)


report = []


def add(x=""):
    print(x)
    report.append(str(x))


# =====================================================
# General
# =====================================================

add("="*70)
add("FINAL RANKING VALIDATION V4")
add("="*70)


add(f"Rows : {len(df)}")
add(f"Cases : {df.case_id.nunique()}")



# =====================================================
# 1) Life expectancy analysis
# =====================================================

add("\n")
add("="*70)
add("1) LIFE EXPECTANCY DISTRIBUTION")
add("="*70)


add(
    df["life_expectancy_years"]
    .value_counts()
    .sort_index()
    .to_string()
)



# =====================================================
# 2) HA validation
# =====================================================

add("\n")
add("="*70)
add("2) HA RELIABILITY TEST")
add("="*70)


ha_df = df[
    df["ha_required"] == 1
]


if len(ha_df) > 0:

    top_ha = (
        ha_df[
            ha_df["relevance_label"] == 3
        ]
    )


    add(
        "Average reliability score label 3:"
    )

    add(
        top_ha["reliability_score"]
        .mean()
    )


    add(
        "Average life expectancy label 3:"
    )

    add(
        top_ha["life_expectancy_years"]
        .mean()
    )


else:

    add("No HA cases found")



# =====================================================
# 3) Preference influence
# =====================================================

add("\n")
add("="*70)
add("3) PREFERENCE INFLUENCE")
add("="*70)



tests = [

    (
        "Performance",
        "performance_priority",
        "performance_score"
    ),

    (
        "Cost",
        "cost_priority",
        "cost_score"
    ),

    (
        "Power",
        "power_priority",
        "power_score"
    ),

    (
        "Reliability",
        "reliability_priority",
        "reliability_score"
    )

]


for name, pref, score in tests:


    high_pref = df[
        df[pref] > 0.7
    ]


    if len(high_pref) == 0:
        add(
            f"{name}: no cases"
        )
        continue


    best = high_pref[
        high_pref["relevance_label"] == 3
    ]


    add(
        f"\n{name} preference"
    )


    add(
        f"Cases: {len(high_pref)}"
    )


    add(
        f"Average {score} of top choices:"
    )


    add(
        best[score]
        .mean()
    )



# =====================================================
# 4) Top label characteristics
# =====================================================

add("\n")
add("="*70)
add("4) TOP RECOMMENDATIONS PROFILE")
add("="*70)


top = df[
    df["relevance_label"] == 3
]


add(
    top[
        [
            "media_type",
            "protocol"
        ]
    ]
    .value_counts()
    .head(10)
    .to_string()
)



# =====================================================
# 5) Missing values
# =====================================================

add("\n")
add("="*70)
add("5) MISSING VALUES")
add("="*70)


missing = (
    df.isnull()
    .sum()
)


missing = missing[
    missing > 0
]


if len(missing)==0:
    add("PASS : no missing values")
else:
    add(missing)



# =====================================================
# Save
# =====================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )


print("\nSaved:")
print(REPORT_FILE)