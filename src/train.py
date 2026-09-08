import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


INPUT_FILE = "data/features.csv"
MODEL_FILE = "models/congestion_model.pkl"


def train_model():

    print("Loading features...")

    df = pd.read_csv(INPUT_FILE)

    # Input features
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

    # Target
    y = df["congestion"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_FILE)

    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_model()
