import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model/fraud_model.joblib")

model = None
try:
    model = load_model()
except Exception as e:
    st.warning("Model file not found. Click 'Use default model' below to load a prebuilt model.")
    st.exception(e)

st.title("🛡️ Fraud Detection (10-Feature)")
st.write("Enter transaction details or upload a CSV to score.")

with st.expander("About the model", expanded=False):
    st.markdown("""
This app uses a scikit-learn **Pipeline** with preprocessing (scaling + one-hot encoding) and a **RandomForestClassifier** trained on synthetic data.
**Features**:
1. transaction_amount
2. transaction_hour
3. user_age
4. account_tenure_days
5. device_trust_score
6. ip_risk_score
7. merchant_category
8. num_prev_transactions_24h
9. chargeback_ratio
10. is_international
    """)

def input_form():
    st.subheader("Single Prediction")
    col1, col2 = st.columns(2)
    with col1:
        transaction_amount = st.number_input("Transaction Amount (USD)", min_value=0.0, value=120.0, step=1.0, format="%.2f")
        transaction_hour = st.number_input("Transaction Hour (0–23)", min_value=0, max_value=23, value=14, step=1)
        user_age = st.number_input("User Age", min_value=18, max_value=99, value=32, step=1)
        account_tenure_days = st.number_input("Account Tenure (days)", min_value=1, max_value=36500, value=400, step=1)
        device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.7, 0.01)
    with col2:
        ip_risk_score = st.slider("IP Risk Score", 0.0, 1.0, 0.2, 0.01)
        merchant_category = st.selectbox("Merchant Category", ['electronics','fashion','grocery','travel','gaming'])
        num_prev_transactions_24h = st.number_input("Prev Transactions (24h)", min_value=0, max_value=1000, value=2, step=1)
        chargeback_ratio = st.slider("Chargeback Ratio", 0.0, 1.0, 0.05, 0.01)
        is_international = st.selectbox("International?", ["No","Yes"])

    data = {
        "transaction_amount": transaction_amount,
        "transaction_hour": transaction_hour,
        "user_age": user_age,
        "account_tenure_days": account_tenure_days,
        "device_trust_score": device_trust_score,
        "ip_risk_score": ip_risk_score,
        "merchant_category": merchant_category,
        "num_prev_transactions_24h": num_prev_transactions_24h,
        "chargeback_ratio": chargeback_ratio,
        "is_international": 1 if is_international=="Yes" else 0,
    }
    return pd.DataFrame([data])

def predict(df):
    proba = model.predict_proba(df)[:,1][0]
    pred = int(proba >= threshold)
    return pred, proba

# Threshold control
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

# Buttons for convenience
if st.sidebar.button("Use default model"):
    try:
        model = load_model()
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

# Single prediction
if model is not None:
    df_single = input_form()
    if st.button("Predict"):
        pred, proba = predict(df_single)
        st.markdown("---")
        st.subheader("Result")
        st.metric("Fraud Probability", f"{proba:.2%}")
        st.write("Prediction:", "**FRAUD**" if pred==1 else "**LEGIT**")
        st.caption(f"Threshold = {threshold:.2f}")

# Batch scoring
st.markdown("---")
st.subheader("Batch Scoring (CSV)")
st.write("Upload a CSV with the **exact 10 columns**. Download a template below.")

template = pd.DataFrame([{
    "transaction_amount": 120.0,
    "transaction_hour": 14,
    "user_age": 32,
    "account_tenure_days": 400,
    "device_trust_score": 0.7,
    "ip_risk_score": 0.2,
    "merchant_category": "electronics",
    "num_prev_transactions_24h": 2,
    "chargeback_ratio": 0.05,
    "is_international": 0,
}])
st.download_button("Download CSV template", data=template.to_csv(index=False), file_name="fraud_template.csv", mime="text/csv")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if model is not None and uploaded is not None:
    try:
        batch = pd.read_csv(uploaded)
        probs = model.predict_proba(batch)[:,1]
        preds = (probs >= threshold).astype(int)
        out = batch.copy()
        out["fraud_probability"] = probs
        out["prediction"] = preds
        st.success(f"Scored {len(out)} rows.")
        st.dataframe(out.head(20))
        st.download_button("Download Scored CSV", data=out.to_csv(index=False), file_name="scored.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to score CSV: {e}")
