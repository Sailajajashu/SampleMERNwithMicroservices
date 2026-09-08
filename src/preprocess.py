import pandas as pd


INPUT_FILE = "data/5g_metrics.csv"
OUTPUT_FILE = "data/clean_data.csv"


def preprocess():

    print("Reading raw 5G data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Raw records: {len(df)}")

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove records with missing values
    df = df.dropna()

    # Validate PRB utilization
    df = df[
        (df["prb_utilization"] >= 0)
        & (df["prb_utilization"] <= 100)
    ]

    # Validate latency
    df = df[df["latency"] >= 0]

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Clean records: {len(df)}")
    print(f"Clean data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess()
