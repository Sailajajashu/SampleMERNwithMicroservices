import os
import sys

import mlflow


MODEL_NAME = "5G-Congestion-Model"


def register_model(run_id):

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    model_uri = f"runs:/{run_id}/model"

    print(
        f"Registering model from:"
    )

    print(model_uri)

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print()
    print(
        "MODEL REGISTERED SUCCESSFULLY"
    )

    print(
        f"Model name : {result.name}"
    )

    print(
        f"Version    : {result.version}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
            " python src/register_model.py <run_id>"
        )

        sys.exit(1)

    register_model(
        sys.argv[1]
    )
