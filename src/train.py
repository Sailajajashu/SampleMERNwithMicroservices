import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "5G-Congestion-Prediction"

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


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train(train_file, model_dir):

    # --------------------------------------------------------
    # Prepare model directory
    # --------------------------------------------------------

    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------------

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    print()
    print("=" * 60)
    print("MLFLOW CONFIGURATION")
    print("=" * 60)

    print(f"Tracking URI : {tracking_uri}")
    print(f"Experiment   : {EXPERIMENT_NAME}")

    # --------------------------------------------------------
    # Create experiment if it doesn't exist
    # --------------------------------------------------------

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:

        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME
        )

        print(
            f"Created MLflow experiment: "
            f"{EXPERIMENT_NAME}"
        )

    else:

        experiment_id = experiment.experiment_id

        print(
            f"MLflow experiment already exists: "
            f"{EXPERIMENT_NAME}"
        )

    print(f"Experiment ID: {experiment_id}")

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # ========================================================
    # LOAD TRAINING DATA
    # ========================================================

    df = pd.read_csv(train_file)

    print()
    print("=" * 60)
    print("5G CONGESTION MODEL TRAINING")
    print("=" * 60)

    print(f"Training data : {train_file}")
    print(f"Rows          : {len(df)}")
    print(f"Columns       : {list(df.columns)}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("ERROR: Missing required columns:")

        for column in missing_columns:
            print(f"  - {column}")

        sys.exit(1)

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X = df[FEATURES]
    y = df[TARGET]

    print()
    print("Features:")

    for feature in FEATURES:
        print(f"  - {feature}")

    print()
    print(f"Target: {TARGET}")

    # ========================================================
    # MODEL HYPERPARAMETERS
    # ========================================================

    n_estimators = 200
    max_depth = 10
    random_state = 42

    # ========================================================
    # START MLFLOW RUN
    # ========================================================

    with mlflow.start_run(
        run_name="5G-Congestion-RandomForest"
    ) as run:

        # ====================================================
        # MLFLOW TAGS
        # ====================================================

        mlflow.set_tag(
            "project",
            "5G-Congestion-Prediction",
        )

        mlflow.set_tag(
            "environment",
            "dev",
        )

        mlflow.set_tag(
            "pipeline",
            "Jenkins",
        )

        mlflow.set_tag(
            "model_type",
            "RandomForest",
        )

        mlflow.set_tag(
            "model_stage",
            "candidate",
        )

        mlflow.set_tag(
            "team",
            "MLOps",
        )

        mlflow.set_tag(
            "framework",
            "scikit-learn",
        )

        mlflow.set_tag(
            "dataset",
            "5G-Network-Metrics",
        )

        # ====================================================
        # TRAIN MODEL
        # ====================================================

        print()
        print("=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        model.fit(
            X,
            y,
        )

        # ====================================================
        # PREDICTIONS
        # ====================================================

        predictions = model.predict(X)

        # ====================================================
        # METRICS
        # ====================================================

        accuracy = accuracy_score(
            y,
            predictions,
        )

        precision = precision_score(
            y,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0,
        )

        # ====================================================
        # SAVE LOCAL MODEL
        # ====================================================

        model_path = model_dir / "model.pkl"

        joblib.dump(
            model,
            model_path,
        )

        # ----------------------------------------------------
        # Verify model was actually created
        # ----------------------------------------------------

        if not model_path.exists():

            raise RuntimeError(
                f"Model file was NOT created: "
                f"{model_path}"
            )

        model_size = model_path.stat().st_size

        print()
        print("=" * 60)
        print("LOCAL MODEL")
        print("=" * 60)

        print(
            f"Model saved : "
            f"{model_path}"
        )

        print(
            f"Model size  : "
            f"{model_size} bytes"
        )

        # ====================================================
        # SAVE FEATURE METADATA
        # ====================================================

        feature_file = (
            model_dir / "features.txt"
        )

        feature_file.write_text(
            "\n".join(FEATURES)
        )

        print(
            f"Features saved : "
            f"{feature_file}"
        )

        # ====================================================
        # MLFLOW PARAMETERS
        # ====================================================

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators,
        )

        mlflow.log_param(
            "max_depth",
            max_depth,
        )

        mlflow.log_param(
            "random_state",
            random_state,
        )

        mlflow.log_param(
            "training_rows",
            len(df),
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURES),
        )

        mlflow.log_param(
            "target_column",
            TARGET,
        )

        # ====================================================
        # MLFLOW METRICS
        # ====================================================

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1",
            f1,
        )

        # ====================================================
        # LOG MODEL TO MLFLOW
        # ====================================================

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
        )

        # ====================================================
        # LOG FEATURE METADATA
        # ====================================================

        mlflow.log_artifact(
            str(feature_file),
            artifact_path="metadata",
        )

        # ====================================================
        # MODEL RESULTS
        # ====================================================

        print()
        print("=" * 60)
        print("MODEL RESULTS")
        print("=" * 60)

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1       : {f1:.4f}"
        )

        # ====================================================
        # MLFLOW INFORMATION
        # ====================================================

        print()
        print("=" * 60)
        print("MLFLOW INFORMATION")
        print("=" * 60)

        print(
            f"Experiment : "
            f"{EXPERIMENT_NAME}"
        )

        print(
            f"Experiment ID : "
            f"{experiment_id}"
        )

        print(
            f"Run ID     : "
            f"{run.info.run_id}"
        )

        print(
            f"Run Name   : "
            f"{run.info.run_name}"
        )

        print(
            f"Tracking   : "
            f"{tracking_uri}"
        )

        print(
            f"Model      : "
            f"{model_path}"
        )

        # ====================================================
        # DISPLAY TAGS
        # ====================================================

        print()
        print("MLflow Tags:")
        print("-" * 60)

        for key, value in run.data.tags.items():

            print(
                f"  {key:15} = {value}"
            )

        # ====================================================
        # FINAL VERIFICATION
        # ====================================================

        print()
        print("=" * 60)
        print("FILE VERIFICATION")
        print("=" * 60)

        print(
            f"model.pkl exists   : "
            f"{model_path.exists()}"
        )

        print(
            f"features.txt exists: "
            f"{feature_file.exists()}"
        )

        # ====================================================
        # COMPLETED
        # ====================================================

        print()
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print()
        print(
            "MLflow Run URL:"
        )

        print(
            f"{tracking_uri}/#/experiments/"
            f"{experiment_id}/runs/"
            f"{run.info.run_id}"
        )

        print()
        print(
            "IMPORTANT - SAVE THIS RUN ID:"
        )

        print(
            run.info.run_id
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print()
        print("Usage:")
        print(
            "python src/train.py "
            "<train_file> <model_dir>"
        )

        print()
        print("Example:")
        print(
            "python src/train.py "
            "data/processed/train_features.csv "
            "models"
        )

        sys.exit(1)

    train_file = sys.argv[1]

    model_dir = sys.argv[2]

    train(
        train_file,
        model_dir,
        )
