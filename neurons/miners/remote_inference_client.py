import requests


class RemoteInferenceClient:
    def __init__(self, base_url, timeout=15.0, token=""):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token

    def predict_batch(self, texts):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        response = requests.post(
            f"{self.base_url}/predict",
            json={"texts": texts},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        predictions = data.get("predictions")
        if not isinstance(predictions, list):
            raise ValueError("Remote inference response did not contain a predictions list")

        if len(predictions) != len(texts):
            raise ValueError(
                f"Remote inference returned {len(predictions)} predictions for {len(texts)} texts"
            )

        return predictions
