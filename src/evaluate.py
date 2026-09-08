import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


INPUT_FILE = "data/features.csv"
MODEL_FILE = "models/congestion_model.pkl"

F1_THRESHOLD = 0.90


def evaluate_model():

    print("Loading data...")

    df = pd.read_csv(INPUT_FILE)

    X = df[
        [
            "users",
            "prb_utilization",
            "latency",
            "packet_loss",
            "users_prb_ratio",
            "latency_packet_loss"
        ]
    ]

    y = df["congestion"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Load trained model
    model = joblib.load(MODEL_FILE)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print()
    print("========== MODEL EVALUATION ==========")
    print(f"Accuracy  : {accuracy:.2f}")
    print(f"Precision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F1 Score  : {f1:.2f}")
    print("======================================")
    print()

    # QUALITY GATE
    if f1 < F1_THRESHOLD:

        print(
            f"MODEL FAILED ❌ "
            f"F1 {f1:.2f} < {F1_THRESHOLD}"
        )

        # Non-zero exit code
        raise SystemExit(1)

    print(
        f"MODEL PASSED ✅ "
        f"F1 {f1:.2f} >= {F1_THRESHOLD}"
    )


if __name__ == "__main__":
    evaluate_model()
