import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "cell_id",
    "users",
    "prb_utilization",
    "latency",
    "packet_loss",
    "congestion",
]


def validate_data(input_file):
    print(f"Validating dataset: {input_file}")

    df = pd.read_csv(input_file)

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Check columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty")

    # Check null values
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()

    if null_counts.sum() > 0:
        print("Null values found:")
        print(null_counts)
        raise ValueError("Dataset contains null values")

    # Check duplicates
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"Dataset contains {duplicates} duplicate rows"
        )

    # Range validation
    if (df["users"] < 0).any():
        raise ValueError("Users cannot be negative")

    if not df["prb_utilization"].between(0, 100).all():
        raise ValueError(
            "PRB utilization must be between 0 and 100"
        )

    if (df["latency"] < 0).any():
        raise ValueError("Latency cannot be negative")

    if not df["packet_loss"].between(0, 100).all():
        raise ValueError(
            "Packet loss must be between 0 and 100"
        )

    if not df["congestion"].isin([0, 1]).all():
        raise ValueError(
            "Congestion must contain only 0 or 1"
        )

    # Target distribution
    print("\nTarget distribution:")
    print(df["congestion"].value_counts())

    print("\nData validation PASSED")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python src/data_validation.py "
            "data/5g_metrics.csv"
        )
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    try:
        validate_data(input_file)
    except Exception as exc:
        print(f"DATA VALIDATION FAILED: {exc}")
        sys.exit(1)
