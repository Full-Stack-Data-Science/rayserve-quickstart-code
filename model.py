import joblib
import numpy as np
import pandas as pd


class HousePricePredictor:
    def __init__(self):
        # Load pre-trained model (Assume it's a scikit-learn regression model)
        self.model = joblib.load("../models/house_price_model.joblib")

    def predict(self, features: list) -> float:
        # Convert input to a DataFrame
        features_df = pd.DataFrame(features)

        # Run inference
        predicted_price = self.model.predict(features_df)[0]

        return predicted_price


# Example usage
if __name__ == "__main__":
    predictor = HousePricePredictor()

    # Example house
    example_features = {
        "Id": {416: 416},
        "MSSubClass": {416: 60},
        "MSZoning": {416: "RL"},
        "LotArea": {416: 7844},
        "LotConfig": {416: "Inside"},
        "BldgType": {416: "1Fam"},
        "OverallCond": {416: 7},
        "YearBuilt": {416: 1978},
        "YearRemodAdd": {416: 1978},
        "Exterior1st": {416: "HdBoard"},
        "BsmtFinSF2": {416: 0.0},
        "TotalBsmtSF": {416: 672.0},
    }

    predicted_price = predictor.predict(example_features)

    print(f"Predicted House Price: {predicted_price}")
