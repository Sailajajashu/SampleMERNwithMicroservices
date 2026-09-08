
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate():

    print("==========================================")
    print("5G CONGESTION MODEL EVALUATION")
    print("==========================================")

    # ==================================================
    # 1. LOAD FEATURE DATA
    # ==================================================

    print("Loading feature data...")

    df = pd.read_csv("data/features.csv")

    print(f"Total records: {len(df)}")

    # ==================================================
    # 2. LOAD TRAINED MODEL
    # ==================================================

    print("Loading trained model...")

    model = joblib.load(
        "models/congestion_model.pkl"
    )

    print("Model loaded successfully.")

    # ==================================================
    # 3. DEFINE FEATURES
    # ==================================================

    features = [
        "users",
        "prb_utilization",
        "latency",
        "packet_loss",
        "users_prb_ratio",
        "latency_packet_loss"
    ]

    target = "congestion"

    # Check required columns
    required_columns = features + [target]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ==================================================
    # 4. PREPARE DATA
    # ==================================================

    X = df[features]

    y = df[target]

    # ==================================================
    # 5. SAME TRAIN/TEST SPLIT AS TRAIN.PY
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("")
    print(f"Training records : {len(X_train)}")
    print(f"Testing records  : {len(X_test)}")

    # ==================================================
    # 6. PREDICT
    # ==================================================

    print("")
    print("Generating predictions...")

    predictions = model.predict(X_test)

    # ==================================================
    # 7. CALCULATE METRICS
    # ==================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    # ==================================================
    # 8. DISPLAY RESULTS
    # ==================================================

    print("")
    print("==========================================")
    print("MODEL EVALUATION RESULTS")
    print("==========================================")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("==========================================")

    # ==================================================
    # 9. SAVE F1 SCORE FOR JENKINS
    # ==================================================

    print("")
    print("Saving F1 score for Jenkins...")

    with open("f1_score.txt", "w") as file:
        file.write(str(f1))

    print("F1 score saved successfully.")
    print("File: f1_score.txt")

    # ==================================================
    # 10. DISPLAY QUALITY INFORMATION
    # ==================================================

    print("")
    print("Model evaluation completed successfully.")

    print("")
    print("Jenkins will use f1_score.txt")
    print("to perform the Model Quality Gate.")

    print("==========================================")


if __name__ == "__main__":
    evaluate()
