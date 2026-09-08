import pandas as pd


def preprocess():

    print("Reading raw 5G data...")

    df = pd.read_csv("data/5g_metrics.csv")

    print(f"Raw records: {len(df)}")

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print("Columns found:")
    print(df.columns.tolist())

    required_columns = [
        "cell_id",
        "users",
        "prb_utilization",
        "latency",
        "packet_loss",
        "congestion"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna()

    # Validate PRB utilization
    df = df[
        (df["prb_utilization"] >= 0) &
        (df["prb_utilization"] <= 100)
    ]

    # Validate latency
    df = df[df["latency"] >= 0]

    # Validate packet loss
    df = df[df["packet_loss"] >= 0]

    df.to_csv(
        "data/clean_data.csv",
        index=False
    )

    print(
        f"Clean records: {len(df)}"
    )

    print(
        "Clean data saved to data/clean_data.csv"
    )


if __name__ == "__main__":
    preprocess()
