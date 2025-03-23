import joblib
import pandas as pd
from ray import serve
from starlette.requests import Request


# Define a deployment with the following configurations
# Number of deployment process: 2, each has 0.2 cpu and 0 gpu
@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class HousePricePredictor:
    def __init__(self):
        # Load pre-trained model (Assume it's a scikit-learn regression model)
        self.model = joblib.load("models/house_price_model.joblib")

    def predict(self, features: dict) -> float:
        # Convert input to a DataFrame
        features_df = pd.DataFrame(features)

        # Run inference
        predicted_price = self.model.predict(features_df)[0]

        return predicted_price

    async def __call__(self, http_request: Request) -> float:
        data: dict = await http_request.json()
        return self.predict(data)


house_price_predictor_app = HousePricePredictor.bind()
