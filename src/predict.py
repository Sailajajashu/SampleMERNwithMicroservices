import joblib
import pandas as pd


MODEL_FILE = "models/congestion_model.pkl"


def predict_congestion():

    print("Loading trained model...")

    model = joblib.load(MODEL_FILE)

    # New 5G network observation
    data = pd.DataFrame([
        {
            "users": 1800,
            "prb_utilization": 88,
            "latency": 38,
            "packet_loss": 1.0
        }
    ])

    # Create same features used during training

    data["users_prb_ratio"] = (
        data["users"] / data["prb_utilization"]
    )

    data["latency_packet_loss"] = (
        data["latency"] * data["packet_loss"]
    )

    features = data[
        [
            "users",
            "prb_utilization",
            "latency",
            "packet_loss",
            "users_prb_ratio",
            "latency_packet_loss"
        ]
    ]

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    print()
    print("========== 5G PREDICTION ==========")

    if prediction == 1:
        print("Prediction: CONGESTION ⚠️")
    else:
        print("Prediction: NORMAL ✅")

    print(
        f"Congestion probability: "
        f"{probability:.2%}"
    )

    print("===================================")


if __name__ == "__main__":
    predict_congestion()
