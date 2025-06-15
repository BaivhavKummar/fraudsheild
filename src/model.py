# src/model.py

from xgboost import XGBClassifier
import joblib
from pathlib import Path

def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, preprocessor, model_path="models/xgb_model.pkl", preprocessor_path="models/preprocessor.pkl"):
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
