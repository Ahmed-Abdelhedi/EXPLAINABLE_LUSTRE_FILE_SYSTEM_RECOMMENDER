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
    "ranking_audit_report_v4.txt"
)



# =====================================================
# Load dataset
# =====================================================

df = pd.read_csv(INPUT_FILE)


report = []


def add(text=""):
    print(text)
    report.append(text)



# =====================================================
# General information
# =====================================================

add("=" * 70)
add("RANKING AUDIT REPORT V4")
add("=" * 70)


add(
    f"Rows : {len(df)}"
)

add(
    f"Cases : {df['case_id'].nunique()}"
)

add(
    f"Columns : {len(df.columns)}"
)



# =====================================================
# 1) Label distribution
# =====================================================

add("\n")
add("=" * 70)
add("1) LABEL DISTRIBUTION")
add("=" * 70)


label_dist = (
    df["relevance_label"]
    .value_counts()
    .sort_index()
)


add(
    str(label_dist)
)



# =====================================================
# 2) Candidates per case
# =====================================================

add("\n")
add("=" * 70)
add("2) CANDIDATES PER CASE")
add("=" * 70)


candidate_stats = (
    df.groupby("case_id")
    .size()
    .describe()
)


add(
    str(candidate_stats)
)



# =====================================================
# 3) Check ranking score ordering
# =====================================================

add("\n")
add("=" * 70)
add("3) LABEL VS RANKING SCORE")
add("=" * 70)



label_score = (
    df.groupby("relevance_label")
    ["ranking_score"]
    .mean()
)


add(
    str(label_score)
)


if (
    label_score.loc[3]
    >
    label_score.loc[2]
    >
    label_score.loc[1]
    >
    label_score.loc[0]
):

    add(
        "PASS: labels follow ranking_score order"
    )

else:

    add(
        "WARNING: label order inconsistent"
    )



# =====================================================
# 4) Performance preference test
# =====================================================


add("\n")
add("=" * 70)
add("4) PERFORMANCE PREFERENCE TEST")
add("=" * 70)



perf_cases = df[
    df["performance_priority"] > 0.7
]


if len(perf_cases) > 0:


    top_perf = (
        perf_cases[
            perf_cases["relevance_label"] == 3
        ]
        [
            [
                "media_type",
                "protocol",
                "seq_read",
                "seq_write"
            ]
        ]
        .describe(include="all")
    )


    add(
        str(top_perf)
    )


else:

    add(
        "No high performance cases found"
    )



# =====================================================
# 5) Cost preference test
# =====================================================


add("\n")
add("=" * 70)
add("5) COST PREFERENCE TEST")
add("=" * 70)



cost_cases = df[
    df["cost_priority"] > 0.7
]


if len(cost_cases) > 0:


    top_cost_price = (
        cost_cases[
            cost_cases["relevance_label"] == 3
        ]
        ["price"]
        .mean()
    )


    add(
        f"Average price of top cost choices: {top_cost_price:.2f}"
    )


else:

    add(
        "No high cost-priority cases found"
    )



# =====================================================
# 6) Power preference test
# =====================================================


add("\n")
add("=" * 70)
add("6) POWER PREFERENCE TEST")
add("=" * 70)



power_cases = df[
    df["power_priority"] > 0.7
]


if len(power_cases) > 0:


    top_power = (
        power_cases[
            power_cases["relevance_label"] == 3
        ]
        ["power"]
        .mean()
    )


    all_power = (
        power_cases["power"]
        .mean()
    )


    add(
        f"Top label power average : {top_power:.2f} W"
    )


    add(
        f"All candidates power average : {all_power:.2f} W"
    )


else:

    add(
        "No high power-priority cases found"
    )



# =====================================================
# 7) Reliability preference test
# =====================================================


add("\n")
add("=" * 70)
add("7) RELIABILITY PREFERENCE TEST")
add("=" * 70)


rel_cases = df[
    df["reliability_priority"] > 0.7
]


if len(rel_cases) > 0:


    top_rel = (
        rel_cases[
            rel_cases["relevance_label"] == 3
        ]
        ["life_expectancy_years"]
        .mean()
    )


    add(
        f"Top reliability life expectancy: {top_rel:.2f} years"
    )


else:

    add(
        "No high reliability cases found"
    )



# =====================================================
# 8) Missing values
# =====================================================

add("\n")
add("=" * 70)
add("8) MISSING VALUES")
add("=" * 70)


missing = (
    df.isnull()
    .sum()
)


missing = missing[
    missing > 0
]


if len(missing) == 0:

    add(
        "PASS: no missing values"
    )

else:

    add(
        str(missing)
    )



# =====================================================
# Save report
# =====================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )


print("\nReport saved:")
print(REPORT_FILE)