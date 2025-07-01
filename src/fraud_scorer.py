# src/fraud_scorer.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path("models/xgb_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

class FraudScorer:
    def __init__(self, model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.expected_features = list(self.preprocessor.feature_names_in_)  # Save expected order

    def score_transaction(self, input_dict):
        """
        Accepts input as a dictionary (e.g., from Streamlit form).
        Returns: predicted class, fraud probability (threat score), threat flag.
        """

        # Fill missing keys with 0 (e.g., 'id') and reorder to match preprocessor
        for feat in self.expected_features:
            if feat not in input_dict:
                input_dict[feat] = 0

        input_df = pd.DataFrame([[input_dict[f] for f in self.expected_features]], columns=self.expected_features)

        X_preprocessed = self.preprocessor.transform(input_df)
        prob = self.model.predict_proba(X_preprocessed)[0][1]  # probability of fraud
        prediction = int(prob > 0.5)
        threat_flag = "🚨 Trigger 2FA" if prob > 0.5 else "✅ Normal"

        return {
            "prediction": prediction,
            "threat_score": round(prob * 100, 2),
            "threat_flag": threat_flag
        }
