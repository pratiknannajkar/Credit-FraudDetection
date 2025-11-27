import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import joblib, json
from pathlib import Path

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detector (KNN Model)")
st.caption("Upload a CSV or enter a single transaction to get a prediction.")

# ---- LOAD MODEL + METADATA ----
MODEL_PATH = "fraud_knn_model.pkl"
META_PATH  = "fraud_threshold.json"

model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

THRESHOLD = float(meta["threshold"])
FEATURES  = meta["features"]

# Try to load local dataset to compute sensible defaults for the single-input form
MEDIANS = None
if Path("creditcard.csv").exists():
    try:
        tmp = pd.read_csv("creditcard.csv", nrows=10000)  # small sample is enough
        MEDIANS = tmp[FEATURES].median(numeric_only=True)
    except Exception:
        pass

# ===============  MODE SELECTOR  ===============
mode = st.radio("Choose mode:", ["Upload CSV", "Single input"], horizontal=True)

# ===============  CSV MODE  ====================
if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (same columns as training)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

        # Ensure all expected columns exist; add missing as zeros
        for c in FEATURES:
            if c not in df.columns:
                df[c] = 0
        df = df[FEATURES]

        proba = model.predict_proba(df)[:, 1]
        pred  = (proba >= THRESHOLD).astype(int)

        out = df.copy()
        out["fraud_probability"] = proba
        out["fraud_pred"] = pred

        st.success(f"Scored {len(out)} rows")
        st.dataframe(out.head(100))
        st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode("utf-8"),
                           "fraud_predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV to score.")

# ===============  SINGLE INPUT MODE  ===========
else:
    st.write("Enter values for one transaction. Leave as defaults if unsure.")
    with st.form("single_tx_form"):
        cols = []
        values = {}
        # Build inputs dynamically for all FEATURES
        # We’ll arrange inputs in rows of 3 to keep it tidy.
        for i, feat in enumerate(FEATURES):
            if i % 3 == 0:
                cols = st.columns(3)
            default = float(MEDIANS.get(feat)) if (MEDIANS is not None and feat in MEDIANS) else 0.0
            # Most Kaggle features (Time, V1..V28, Amount) are numeric; use number_input
            values[feat] = cols[i % 3].number_input(feat, value=default, format="%.6f")

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([values])[FEATURES]
        proba = model.predict_proba(row)[:, 1][0]
        pred  = int(proba >= THRESHOLD)

        st.subheader("Result")
        st.metric(label="Predicted class (0=Legit, 1=Fraud)", value=str(pred))
        st.progress(min(max(proba, 0.0), 1.0))
        st.write(f"**Fraud probability:** {proba:.4f}  |  **Threshold:** {THRESHOLD:.4f}")

        st.download_button("⬇️ Download this single prediction",
                           row.assign(fraud_probability=proba, fraud_pred=pred).to_csv(index=False).encode("utf-8"),
                           "single_prediction.csv", "text/csv")
