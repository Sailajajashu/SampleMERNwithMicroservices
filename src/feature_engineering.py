import sys
from pathlib import Path

import pandas as pd


TARGET = "congestion"


def engineer_features(input_file, output_file):

    df = pd.read_csv(input_file)

    # Risk-oriented derived features
    df["user_prb_interaction"] = (
        df["users"] * df["prb_utilization"]
    )

    df["latency_packet_loss"] = (
        df["latency"] * df["packet_loss"]
    )

    df["high_prb_flag"] = (
        df["prb_utilization"] >= 80
    ).astype(int)

    df["high_latency_flag"] = (
        df["latency"] >= 50
    ).astype(int)

    df.to_csv(output_file, index=False)

    print(f"Feature engineered dataset saved to {output_file}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python src/feature_engineering.py "
            "<input> <output>"
        )
        sys.exit(1)

    engineer_features(
        sys.argv[1],
        sys.argv[2],
    )
