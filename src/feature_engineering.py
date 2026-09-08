import pandas as pd


INPUT_FILE = "data/clean_data.csv"
OUTPUT_FILE = "data/features.csv"


def create_features():

    print("Creating ML features...")

    df = pd.read_csv(INPUT_FILE)

    # Create useful telecom features

    df["users_prb_ratio"] = (
        df["users"] / df["prb_utilization"]
    )

    df["latency_packet_loss"] = (
        df["latency"] * df["packet_loss"]
    )

    # Select final features

    features = df[
        [
            "users",
            "prb_utilization",
            "latency",
            "packet_loss",
            "users_prb_ratio",
            "latency_packet_loss",
            "congestion"
        ]
    ]

    features.to_csv(OUTPUT_FILE, index=False)

    print("Features created successfully.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_features()
