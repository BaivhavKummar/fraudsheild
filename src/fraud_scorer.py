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

    def score_transaction(self, input_dict):
        """
        Accepts input as a dictionary (e.g., from Streamlit form).
        Returns: predicted class, fraud probability (threat score), threat flag.
        """
        input_df = pd.DataFrame([input_dict])
        X_preprocessed = self.preprocessor.transform(input_df)
        
        prob = self.model.predict_proba(X_preprocessed)[0][1]  # probability of fraud
        prediction = int(prob > 0.5)
        threat_flag = "🚨 Trigger 2FA" if prob > 0.5 else "✅ Normal"

        return {
            "prediction": prediction,
            "threat_score": round(prob * 100, 2),
            "threat_flag": threat_flag
        }

