# 🛡️ FraudSheild: Credit Card Fraud Detection & Explanation

FraudSheild is a machine learning pipeline that detects fraudulent credit card transactions using XGBoost and provides transparent explanations using SHAP. The solution is packaged with a web app powered by Streamlit for real-time scoring and interpretability.

---

## 🚀 Project Overview

This repository includes:

- ✅ Clean data preprocessing pipeline
- ✅ Model training using XGBoost
- ✅ Model interpretation using SHAP
- ✅ Deployment-ready Streamlit app
- ✅ Modular Python code and notebooks

---

## 🧠 How to Use the Credit Card Fraud Detection App

This app uses a trained machine learning model with SHAP-based explainability to analyze and detect whether a credit card transaction is fraudulent based on its features.

### 🔍 Transaction Input Instructions

You need to fill in the following **30 numerical fields** in the sidebar:

* `Time`: Time (in seconds) since the start of the dataset
* `Amount`: Value of the transaction
* `V1` to `V28`: Anonymized features created using PCA (no direct meaning)

---

### ✅ **Simulate a Normal Transaction**

Use these values to simulate a typical, **non-fraudulent** transaction:

| Feature | Suggested Value               | Notes                                  |
| ------- | ----------------------------- | -------------------------------------- |
| Time    | `36000`                       | Morning or daytime transaction (10 AM) |
| Amount  | `45.00`                       | Small grocery-like purchase            |
| V1      | `-0.5`                        | Typical low-magnitude values           |
| V2      | `0.4`                         |                                        |
| V3      | `-0.2`                        |                                        |
| V4      | `0.1`                         |                                        |
| V5      | `0.0`                         |                                        |
| V6–V28  | `Values between -1.0 and 1.0` | Random but close to 0                  |

🟢 **Expected Output**:

* Threat Score: Low
* Prediction: ✅ Not Fraud
* Flag: ✅ Normal

---

### 🚨 **Simulate a Fraudulent Transaction**

Use these values to simulate a likely **fraudulent** transaction:

| Feature | Suggested Value                | Notes                              |
| ------- | ------------------------------ | ---------------------------------- |
| Time    | `86400`                        | Late-night or odd-time transaction |
| Amount  | `1500.00`                      | High-value transaction             |
| V1      | `-3.2`                         | Unusually high-magnitude V values  |
| V2      | `4.5`                          |                                    |
| V3      | `-2.7`                         |                                    |
| V4      | `1.8`                          |                                    |
| V5      | `-4.1`                         |                                    |
| V6–V28  | `Values between -3.0 and +5.0` | Skewed or abnormal patterns        |

🔴 **Expected Output**:

* Threat Score: High
* Prediction: ❌ Fraud
* Flag: 🚨 Trigger 2FA

---

### 💡 Tips

* For custom analysis, modify the values across V1–V28 based on your test patterns.
* All fields are required. Keep decimal values (e.g., `0.00`) for precision.
* Press the **⚡ Analyze Transaction** button once inputs are filled.

---


