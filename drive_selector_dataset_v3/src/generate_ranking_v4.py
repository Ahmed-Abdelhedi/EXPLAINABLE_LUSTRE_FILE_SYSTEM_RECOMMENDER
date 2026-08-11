import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent


INPUT_FILE = (
    BASE_DIR /
    "output" /
    "drive_selector_dataset_v4.csv"
)


OUTPUT_FILE = (
    BASE_DIR /
    "output" /
    "drive_selector_dataset_v4_ranked.csv"
)



# =====================================================
# Load dataset
# =====================================================

df = pd.read_csv(INPUT_FILE)


print("=" * 60)
print("RANKING DATASET GENERATION")
print("=" * 60)

print(
    "Input rows:",
    len(df)
)



# =====================================================
# Helper functions
# =====================================================


def minmax(series):

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        series - min_value
    ) / (
        max_value - min_value
    )



# =====================================================
# 1) Performance score
# =====================================================


print("\nComputing performance score...")


df["seq_read_norm"] = (
    minmax(
        pd.to_numeric(
            df["seq_read"],
            errors="coerce"
        )
        .fillna(0)
    )
)


df["seq_write_norm"] = (
    minmax(
        pd.to_numeric(
            df["seq_write"],
            errors="coerce"
        )
        .fillna(0)
    )
)


df["read_iops_norm"] = (
    minmax(
        pd.to_numeric(
            df["random_read_iops"],
            errors="coerce"
        )
        .fillna(0)
    )
)


df["write_iops_norm"] = (
    minmax(
        pd.to_numeric(
            df["random_write_iops"],
            errors="coerce"
        )
        .fillna(0)
    )
)



df["performance_score"] = (

    0.40 *
    df["seq_read_norm"]

    +

    0.30 *
    df["seq_write_norm"]

    +

    0.15 *
    df["read_iops_norm"]

    +

    0.15 *
    df["write_iops_norm"]

)



# =====================================================
# 2) Cost score
# Lower price = better
# =====================================================


print("Computing cost score...")


df["price_numeric"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)


df["price_numeric"] = (
    df["price_numeric"]
    .fillna(
        df["price_numeric"].median()
    )
)


df["cost_score"] = (
    1 -
    minmax(
        df["price_numeric"]
    )
)



# =====================================================
# 3) Power score
# Lower consumption = better
# =====================================================


print("Computing power score...")


df["power_numeric"] = pd.to_numeric(
    df["power"],
    errors="coerce"
)


df["power_numeric"] = (
    df["power_numeric"]
    .fillna(
        df["power_numeric"].median()
    )
)


df["power_score"] = (
    1 -
    minmax(
        df["power_numeric"]
    )
)



# =====================================================
# 4) Reliability score
# Handle life_expectancy text
# =====================================================


print("Computing reliability score...")


print(
    "Example life_expectancy values:"
)

print(
    df["life_expectancy"]
    .head()
)



# Extract years from text
# Examples:
# "5 years" -> 5
# "3-year warranty" -> 3

df["life_expectancy_years"] = (

    df["life_expectancy"]
    .astype(str)
    .str.extract(
        r"(\d+)"
    )[0]

)


df["life_expectancy_years"] = pd.to_numeric(
    df["life_expectancy_years"],
    errors="coerce"
)



# Missing values

df["life_expectancy_years"] = (
    df["life_expectancy_years"]
    .fillna(
        df["life_expectancy_years"]
        .median()
    )
)



df["reliability_score"] = (
    minmax(
        df["life_expectancy_years"]
    )
)



# =====================================================
# 5) Personalized ranking score
# =====================================================


print(
    "Computing personalized ranking..."
)


df["ranking_score"] = (

    df["performance_priority"]
    *
    df["performance_score"]

    +

    df["cost_priority"]
    *
    df["cost_score"]

    +

    df["power_priority"]
    *
    df["power_score"]

    +

    df["reliability_priority"]
    *
    df["reliability_score"]

)



# =====================================================
# 6) Ranking inside each user case
# =====================================================


print(
    "Creating ranking groups..."
)


df["rank_position"] = (

    df.groupby("case_id")
    ["ranking_score"]
    .rank(
        ascending=False,
        method="first"
    )

)


df["candidate_count"] = (

    df.groupby("case_id")
    ["case_id"]
    .transform("count")

)



# =====================================================
# 7) Convert ranking to ML labels
# =====================================================


def ranking_label(row):

    ratio = (
        row["rank_position"]
        /
        row["candidate_count"]
    )


    if ratio <= 0.20:
        return 3

    elif ratio <= 0.50:
        return 2

    elif ratio <= 0.80:
        return 1

    else:
        return 0



print(
    "Generating relevance labels..."
)


df["relevance_label"] = (
    df.apply(
        ranking_label,
        axis=1
    )
)



# =====================================================
# Remove temporary columns
# =====================================================


df.drop(
    columns=[
        "seq_read_norm",
        "seq_write_norm",
        "read_iops_norm",
        "write_iops_norm",
        "price_numeric",
        "power_numeric"
    ],
    inplace=True
)



# =====================================================
# Save
# =====================================================


df.to_csv(
    OUTPUT_FILE,
    index=False
)



print("\n" + "=" * 60)

print(
    "Output rows:",
    len(df)
)


print(
    "Output columns:",
    len(df.columns)
)


print(
    "\nLabel distribution:"
)


print(
    df["relevance_label"]
    .value_counts()
    .sort_index()
)



print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print("=" * 60)