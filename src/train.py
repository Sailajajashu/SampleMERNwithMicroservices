import mlflow
import mlflow.sklearn

# MLflow configuration
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("5G-Congestion-Prediction")

# Start MLflow run
with mlflow.start_run():

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Save model locally
    joblib.dump(
        model,
        "models/congestion_model.pkl"
    )

    # Log parameters
    mlflow.log_param("algorithm", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    # Register model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="5G-Congestion-Model"
    )
