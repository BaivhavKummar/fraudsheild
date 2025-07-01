# app/streamlit_app.py

import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import sys, os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.fraud_scorer import FraudScorer

# Load model & explainer
scorer = FraudScorer()
explainer = joblib.load("models/shap_explainer.pkl")

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="wide")
st.title("💳 FraudShield - Real-Time Credit Card Fraud Detection")
st.markdown("Get real-time fraud insights with SHAP explainability and 2FA trigger.")

# Sidebar input
st.sidebar.header("🔍 Transaction Input")

features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
input_data = {}

# Auto-fill buttons
if st.sidebar.button("🔁 Fill Normal Transaction"):
    input_data = {
        'Time': 36000, 'Amount': 45.00,
        **{f'V{i}': round((-1)**i * 0.5, 2) for i in range(1, 29)}
    }
    st.sidebar.success("Filled with normal transaction.")
elif st.sidebar.button("🚨 Fill Fraudulent Transaction"):
    input_data = {
        'Time': 86400, 'Amount': 1500.00,
        **{f'V{i}': round((-1)**i * i * 0.8, 2) for i in range(1, 29)}
    }
    st.sidebar.warning("Filled with fraudulent transaction.")

# Manual inputs
for feature in features:
    input_data[feature] = st.sidebar.number_input(
        feature, 
        value=float(input_data.get(feature, 0.00)),  # 👈 force float
        step=0.01,
        format="%.2f"
    )
# Add required 'id'
input_data["id"] = 0

# Analyze single transaction
if st.button("⚡ Analyze Transaction"):
    with st.spinner("Scoring..."):
        result = scorer.score_transaction(input_data)

        st.success(f"**Prediction:** {'Fraud ❌' if result['prediction'] else 'Not Fraud ✅'}")
        st.metric(label="Threat Score (%)", value=result["threat_score"])
        st.warning(result["threat_flag"] if result["prediction"] else "✅ No threat detected")

        # SHAP Waterfall plot
        st.subheader("📊 Explanation: Why this score?")
        input_df = pd.DataFrame([input_data])
        input_df = input_df[scorer.expected_features]  # reorder for preprocessor
        preprocessed = scorer.preprocessor.transform(input_df)
        shap_values = explainer(preprocessed)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight')

# CSV Batch Upload
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV of Transactions", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'id' not in df.columns:
        df['id'] = 0
    try:
        preds = df.apply(lambda row: scorer.score_transaction(row.to_dict()), axis=1, result_type='expand')
        df['Prediction'] = preds['prediction']
        df['Threat Score'] = preds['threat_score']
        df['Threat Flag'] = preds['threat_flag']
        st.subheader("📊 Batch Prediction Results")
        st.dataframe(df)
    except Exception as e:
        st.error(f"❌ Error during batch processing: {e}")
