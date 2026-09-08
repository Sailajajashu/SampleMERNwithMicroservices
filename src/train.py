
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ==================================================
# MLflow Configuration
# ==================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "5G-Congestion-Prediction"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ==================================================
# Load Feature Data
# ==================================================

print("==========================================")
print("5G CONGESTION MODEL TRAINING")
print("==========================================")

print("Loading feature data...")

df = pd.read_csv("data/features.csv")

print(f"Total records: {len(df)}")


# ==================================================
# Features and Target
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


X = df[features]
y = df[target]


# ==================================================
# Train/Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ==================================================
# Start MLflow Run
# ==================================================

with mlflow.start_run(
    run_name=f"Jenkins-Build-{os.getenv('BUILD_NUMBER', 'local')}"
):

    print("")
    print("Training Random Forest model...")

    # ==================================================
    # Create Model
    # ==================================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # ==================================================
    # Train Model
    # ==================================================

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")


    # ==================================================
    # Save Model Locally
    # ==================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = "models/congestion_model.pkl"

    joblib.dump(
        model,
        model_path
    )

    print("")
    print(f"Model saved to: {model_path}")


    # ==================================================
    # MLflow Parameters
    # ==================================================

    mlflow.log_param(
        "algorithm",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "jenkins_build",
        os.getenv("BUILD_NUMBER", "local")
    )

    mlflow.log_param(
        "features",
        ",".join(features)
    )


    # ==================================================
    # MLflow Tags
    # ==================================================

    mlflow.set_tag(
        "project",
        "5G Congestion Prediction"
    )

    mlflow.set_tag(
        "pipeline",
        "Jenkins"
    )

    mlflow.set_tag(
        "environment",
        "CI"
    )


    # ==================================================
    # Register Model with MLflow
    # ==================================================

    print("")
    print("Logging model to MLflow...")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="5G-Congestion-Model"
    )

    print("")
    print("==========================================")
    print("MODEL LOGGED TO MLFLOW")
    print("==========================================")

    print(
        f"Experiment: {MLFLOW_EXPERIMENT_NAME}"
    )

    print(
        f"Tracking URI: {MLFLOW_TRACKING_URI}"
    )

    print(
        f"Model: 5G-Congestion-Model"
    )

    print(
        f"Run ID: {mlflow.active_run().info.run_id}"
    )

    print("==========================================")


print("")
print("Training completed successfully.")
