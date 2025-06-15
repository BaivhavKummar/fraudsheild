# app/streamlit_app.py

import streamlit as st
import pandas as pd
from src.fraud_scorer import FraudScorer
import shap
import joblib
import matplotlib.pyplot as plt

# Load explainer and scorer
explainer = joblib.load("models/shap_explainer.pkl")
scorer = FraudScorer()

st.set_page_config(page_title="FraudShield", page_icon="🛡️")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Get real-time fraud insights with SHAP-based explainability and a smart 2FA trigger.")

# Define input fields
st.sidebar.header("🔍 Transaction Input")

input_data = {}
for col in ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]:
    input_data[col] = st.sidebar.number_input(f"{col}", value=0.0)

# Score transaction
if st.button("⚡ Analyze Transaction"):
    with st.spinner("Scoring..."):
        result = scorer.score_transaction(input_data)
        
        st.success(f"**Prediction:** {'Fraud' if result['prediction'] else 'Not Fraud'}")
        st.metric(label="Threat Score (%)", value=result["threat_score"])
        st.warning(result["threat_flag"] if result["prediction"] else "No threat detected")

        # SHAP Explanation
        st.subheader("📊 Explanation: Why this score?")
        input_df = pd.DataFrame([input_data])
        preprocessed = scorer.preprocessor.transform(input_df)
        shap_values = explainer(preprocessed)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight')
