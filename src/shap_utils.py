# src/shap_utils.py

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def explain_model_shap(model, X_sample, max_display=10):
    """
    Generate SHAP summary plot and return shap values.

    Args:
        model: trained model (e.g. XGBClassifier)
        X_sample: subset of data to explain
        max_display: how many features to show

    Returns:
        explainer, shap_values
    """
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_sample, max_display=max_display)
    return explainer, shap_values

def get_top_shap_features(shap_values, feature_names, top_n=5):
    """
    Return top N features by mean absolute SHAP value.

    Args:
        shap_values: SHAP values from explainer
        feature_names: list of column names
        top_n: number of top features to return

    Returns:
        top_features_df: pd.DataFrame with feature and importance
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [(feature_names[i], mean_abs_shap[i]) for i in top_idx]

    top_features_df = pd.DataFrame(top_features, columns=['Feature', 'Mean |SHAP|'])
    return top_features_df

