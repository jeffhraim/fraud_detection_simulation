# Fraud Detection (10-Feature) — Streamlit + scikit-learn

End-to-end demo:
- **train_model.py** generates synthetic data and trains a Pipeline (preprocessing + RandomForest), saving to `model/fraud_model.joblib`.
- **app.py** is a Streamlit GUI with 10 inputs + batch CSV scoring.
- **requirements.txt** lists Python deps.

## Features (10)
1. transaction_amount (float)
2. transaction_hour (0-23 int)
3. user_age (int)
4. account_tenure_days (int)
5. device_trust_score (0-1 float)
6. ip_risk_score (0-1 float)
7. merchant_category (categorical: electronics, fashion, grocery, travel, gaming)
8. num_prev_transactions_24h (int)
9. chargeback_ratio (0-1 float)
10. is_international (0/1)

## Quickstart (Local)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train model (also creates data/synthetic_fraud.csv)
python train_model.py

# Run the app
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (via GitHub)
1. Push this folder to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Select your repo, choose **`app.py`** as the entry point, and set the Python version (3.10+ is fine).
4. Add `requirements.txt` under dependencies. The app loads `model/fraud_model.joblib` from the repo.

## Batch Scoring
Use the template download in the app to format your CSV correctly, then upload for predictions. A scored file with probabilities and labels will be downloadable.

## Notes
- The dataset is synthetic for demo purposes—**do not** use in production without retraining on real data.
- To retrain, change `make_synthetic()` or load your real dataset in `train_model.py` and re-run.
