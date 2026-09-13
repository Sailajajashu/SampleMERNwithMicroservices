

import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

FEATURES = [
    "users",
    "prb_utilization",
    "latency",
    "packet_loss",
    "user_prb_interaction",
    "latency_packet_loss",
    "high_prb_flag",
    "high_latency_flag",
]

TARGET = "congestion"


def evaluate(model_file, test_file, output_file, threshold=0.90):

    model = joblib.load(model_file)

    df = pd.read_csv(test_file)

    X = df[FEATURES]
    y = df[TARGET]

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    Path(output_file).write_text(f"{f1:.6f}\n")

    print()
    print(f"F1 saved to {output_file}")

    if f1 < threshold:
        print(
            f"QUALITY GATE FAILED: "
            f"F1 {f1:.4f} < {threshold}"
        )
        sys.exit(1)

    print(
        f"QUALITY GATE PASSED: "
        f"F1 {f1:.4f} >= {threshold}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(
            "Usage: python src/evaluate.py "
            "<model> <test> <output> <threshold>"
        )
        sys.exit(1)

    evaluate(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        float(sys.argv[4]),
    )
