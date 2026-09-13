import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


FEATURES = [
    "users",
    "prb_utilization",
    "latency",
    "packet_loss",
]

TARGET = "congestion"


def preprocess(input_file, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    print(f"Input rows: {len(df)}")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows containing required null values
    df = df.dropna(
        subset=FEATURES + [TARGET]
    )

    X = df[FEATURES]
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Scale numerical features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=FEATURES,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=FEATURES,
    )

    # Save processed datasets
    train_data = X_train_scaled.copy()
    train_data[TARGET] = y_train.reset_index(drop=True)

    test_data = X_test_scaled.copy()
    test_data[TARGET] = y_test.reset_index(drop=True)

    train_file = output_dir / "train.csv"
    test_file = output_dir / "test.csv"
    scaler_file = output_dir / "scaler.pkl"

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    joblib.dump(scaler, scaler_file)

    print(f"Saved: {train_file}")
    print(f"Saved: {test_file}")
    print(f"Saved: {scaler_file}")

    print("Preprocessing completed successfully")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python src/preprocess.py "
            "<input_file> <output_dir>"
        )
        sys.exit(1)

    preprocess(
        sys.argv[1],
        sys.argv[2],
    )
