import logging

import litellm
import mlflow

MLFLOW_TRACKING_URI = "https://alex-mlflow.staging.cloudinfra.zenml.io/"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("tracing test")

logger = logging.getLogger("mlflow")
logger.setLevel(logging.DEBUG)


@mlflow.trace
def call_litellm():
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hey! how's it going?"}],
    )
    return response


if __name__ == "__main__":
    # Call OpenAI API via LiteLLM
    response = call_litellm()

    print(response)
