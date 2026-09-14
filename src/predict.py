```python
import os
import time

import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "5G-Congestion-Model",
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

MODEL_URI = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)


# ============================================================
# LOAD CHAMPION MODEL FROM MLFLOW
# ============================================================

print("=" * 60)
print("5G CONGESTION MODEL")
print("=" * 60)

print(
    f"MLflow Tracking URI : "
    f"{MLFLOW_TRACKING_URI}"
)

print(
    f"Model Name          : "
    f"{MODEL_NAME}"
)

print(
    f"Model Alias         : "
    f"{MODEL_ALIAS}"
)

print(
    f"Model URI           : "
    f"{MODEL_URI}"
)

print()
print("Loading Champion model from MLflow...")


try:

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    print(
        "Champion model loaded successfully"
    )

except Exception as error:

    print()
    print(
        "ERROR: Failed to load Champion "
        "model from MLflow"
    )

    print(
        f"Reason: {error}"
    )

    raise


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="5G Congestion Prediction API",
    description=(
        "5G network congestion prediction "
        "using MLflow Champion model"
    ),
    version="1.0.0",
)


# ============================================================
# PROMETHEUS METRICS
# ============================================================

# Total number of prediction requests
prediction_counter = Counter(
    "prediction_requests_total",
    "Total prediction requests",
)


# Prediction request latency
prediction_latency = Histogram(
    "prediction_request_duration_seconds",
    "Prediction request duration",
)


# ============================================================
# MODEL PREDICTION DISTRIBUTION
# ============================================================

# Tracks how many predictions are 0 and 1
#
# Example Prometheus output:
#
# model_predictions_total{prediction="0"} 100
# model_predictions_total{prediction="1"} 50
#
model_predictions_total = Counter(
    "model_predictions_total",
    "Total model predictions by prediction class",
    ["prediction"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    users: float

    prb_utilization: float

    latency: float

    packet_loss: float


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "application": (
            "5G Congestion Prediction API"
        ),
        "version": "1.0.0",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_uri": MODEL_URI,
        "status": "running",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model")
def model_info():

    return {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_uri": MODEL_URI,
        "mlflow_tracking_uri": (
            MLFLOW_TRACKING_URI
        ),
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest,
):

    # --------------------------------------------------------
    # Start Timer
    # --------------------------------------------------------

    start_time = time.time()

    # --------------------------------------------------------
    # Count Prediction Request
    # --------------------------------------------------------

    prediction_counter.inc()

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    user_prb_interaction = (
        request.users
        * request.prb_utilization
    )

    latency_packet_loss = (
        request.latency
        * request.packet_loss
    )

    high_prb_flag = int(
        request.prb_utilization >= 80
    )

    high_latency_flag = int(
        request.latency >= 50
    )

    features = [[

        request.users,

        request.prb_utilization,

        request.latency,

        request.packet_loss,

        user_prb_interaction,

        latency_packet_loss,

        high_prb_flag,

        high_latency_flag,

    ]]

    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        features
    )[0]

    # --------------------------------------------------------
    # Prediction Probability
    # --------------------------------------------------------

    probability = None

    if hasattr(
        model,
        "predict_proba",
    ):

        probability = float(
            model.predict_proba(
                features
            )[0][1]
        )

    # --------------------------------------------------------
    # MODEL PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    prediction_label = str(
        int(prediction)
    )

    model_predictions_total.labels(
        prediction=prediction_label
    ).inc()

    # --------------------------------------------------------
    # Request Duration
    # --------------------------------------------------------

    duration = (
        time.time() - start_time
    )

    prediction_latency.observe(
        duration
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "congestion": int(
            prediction
        ),

        "congested": bool(
            prediction
        ),

        "probability": probability,

        "model_name": MODEL_NAME,

        "model_alias": MODEL_ALIAS,

    }


# ============================================================
# PROMETHEUS METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain",
    )
```
