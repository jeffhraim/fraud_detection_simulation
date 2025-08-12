#!/usr/bin/env python3
"""
Train a simple fraud detection model with 10 features on synthetic data.
Saves a scikit-learn Pipeline to model/fraud_model.joblib and metrics to model/metrics.json.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import shuffle
import joblib
rng = np.random.default_rng(42)

# 10 features:
# 1. transaction_amount (float)
# 2. transaction_hour (int 0-23)
# 3. user_age (int)
# 4. account_tenure_days (int)
# 5. device_trust_score (float 0-1)
# 6. ip_risk_score (float 0-1)
# 7. merchant_category (categorical: ['electronics','fashion','grocery','travel','gaming'])
# 8. num_prev_transactions_24h (int)
# 9. chargeback_ratio (float 0-1)
# 10. is_international (0/1)

def make_synthetic(n=20000, random_state=42):
    rng = np.random.default_rng(random_state)
    merchant_categories = np.array(['electronics','fashion','grocery','travel','gaming'])
    data = {
        "transaction_amount": rng.lognormal(mean=3.2, sigma=0.7, size=n).round(2),  # skewed
        "transaction_hour": rng.integers(0, 24, size=n),
        "user_age": rng.integers(18, 80, size=n),
        "account_tenure_days": rng.integers(1, 3650, size=n),
        "device_trust_score": rng.random(size=n),
        "ip_risk_score": rng.random(size=n),
        "merchant_category": rng.choice(merchant_categories, size=n, p=[0.25,0.25,0.2,0.2,0.1]),
        "num_prev_transactions_24h": rng.integers(0, 25, size=n),
        "chargeback_ratio": rng.beta(0.7, 6, size=n),  # mostly small ratios
        "is_international": rng.integers(0,2,size=n)
    }
    df = pd.DataFrame(data)
    # Hidden pattern for fraud probability (logit-style)
    # higher amount, late night hours, low device trust, high ip risk, travel/gaming, high prev txs, high chargeback, international
    cat_weight = df["merchant_category"].map({
        "electronics": 0.0, "fashion": 0.2, "grocery": -0.2, "travel": 0.6, "gaming": 0.8
    }).values
    score = (
        0.0025*df["transaction_amount"].values +
        ((df["transaction_hour"].between(0,5)).astype(float) * 0.6) +
        (-0.006*(df["user_age"].values-30)) +
        (-0.0003*df["account_tenure_days"].values) +
        (-1.2*df["device_trust_score"].values) +
        (1.6*df["ip_risk_score"].values) +
        (0.7*cat_weight) +
        (0.05*df["num_prev_transactions_24h"].values) +
        (2.4*df["chargeback_ratio"].values) +
        (0.8*df["is_international"].values) +
        rng.normal(0, 0.4, size=n)
    )
    # Convert score to probability via logistic
    prob = 1/(1+np.exp(-score))
    y = (prob > 0.5).astype(int)  # label
    df["is_fraud"] = y
    return shuffle(df, random_state=random_state).reset_index(drop=True)

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    df = make_synthetic(n=20000, random_state=42)
    df.to_csv("data/synthetic_fraud.csv", index=False)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    numeric_features = ["transaction_amount","transaction_hour","user_age","account_tenure_days",
                        "device_trust_score","ip_risk_score","num_prev_transactions_24h",
                        "chargeback_ratio","is_international"]
    categorical_features = ["merchant_category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = float(roc_auc_score(y_test, y_proba))

    metrics = {"roc_auc": auc, "classification_report": report}
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, "model/fraud_model.joblib")
    print("Saved model to model/fraud_model.joblib")
    print("ROC-AUC:", auc)

if __name__ == "__main__":
    main()
