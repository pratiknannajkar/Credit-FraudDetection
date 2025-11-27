# app_simple.py (improved)
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path
import matplotlib.pyplot as plt

# ============ PAGE SETUP ============
st.set_page_config(page_title="Simple Fraud Checker", page_icon="💳", layout="centered")
st.markdown("<h1 style='text-align:center;'>💳 Simple Credit Card Fraud Checker</h1>", unsafe_allow_html=True)
st.caption("Enter only the best features; remaining features are auto-filled with medians. Uses your trained KNN model.")

# ============ FILE PATHS ============
MODEL_PATH  = "fraud_knn_model.pkl"
THRESH_PATH = "fraud_threshold.json"
BEST_PATH   = "best_features.json"
DATA_PATH   = "creditcard.csv"   # optional (enables better slider ranges & demo)

# ============ LOAD ARTIFACTS ============
model = joblib.load(MODEL_PATH)
with open(THRESH_PATH, "r") as f:
    THRESHOLD_SAVED = float(json.load(f)["threshold"])
with open(BEST_PATH, "r") as f:
    bmeta = json.load(f)
BEST_FEATURES = bmeta["best_features"]      # e.g., ['V17','V14','V10','V12','V11','V16']
MEDIANS       = bmeta["medians"]            # all training features -> median (for autofill)
ALL_FEATURES  = list(MEDIANS.keys())

# ============ FRIENDLY LABELS (UI only) ============
FRIENDLY_NAMES = {
    "V17": "Transaction Risk Factor 1",
    "V14": "Transaction Risk Factor 2",
    "V10": "Device Risk Index",
    "V12": "Customer Spending Pattern",
    "V11": "Transaction Frequency Score",
    "V16": "Merchant Risk Level",
}

# ============ OPTIONAL: LOAD DATA FOR STATS/RANGES ============
df_stats = None
if Path(DATA_PATH).exists():
    try:
        dtmp = pd.read_csv(DATA_PATH, nrows=80000)  # sample for speed
        # ensure expected columns exist
        for c in ALL_FEATURES:
            if c not in dtmp.columns:
                dtmp[c] = 0.0
        df_stats = dtmp[ALL_FEATURES].copy()
    except Exception:
        df_stats = None

def pct(x, p):
    return np.nanpercentile(x, p) if len(x) else 0.0

def get_bounds(name: str):
    """Return (min,max,default) for a feature slider."""
    default = float(MEDIANS.get(name, 0.0))
    if df_stats is not None and name in df_stats.columns:
        lo = float(pct(df_stats[name], 1))
        hi = float(pct(df_stats[name], 99))
        # if collapsed, widen a bit
        if lo == hi:
            lo, hi = default - 1.0, default + 1.0
        return lo, hi, default
    # fallback
    return default - 1.0, default + 1.0, default

# ============ SIDEBAR: THRESHOLD CONTROL ============
st.sidebar.header("Decision Threshold")
thr = st.sidebar.slider(
    "Choose decision threshold (higher = fewer false alarms, lower = higher recall)",
    min_value=0.01, max_value=0.99, value=float(THRESHOLD_SAVED), step=0.005
)
st.sidebar.caption(f"Saved training threshold: **{THRESHOLD_SAVED:.3f}**")

# ============ PRESET BUTTONS (fill likely values quickly) ============
st.markdown("### Presets")
c1, c2 = st.columns(2)
use_legit = c1.button("Fill Likely Legit")
use_fraud = c2.button("Fill Likely Fraud")

# ============ SINGLE TRANSACTION INPUT ============
st.markdown("## Single Transaction Input")

with st.form("one_tx_form"):
    cols = st.columns(3)
    user_vals = {}

    for i, feat in enumerate(BEST_FEATURES):
        label = FRIENDLY_NAMES.get(feat, feat)
        lo, hi, default = get_bounds(feat)

        # presets adjust default before showing
        if use_legit:
            # nudge towards median (low risk)
            default = float(MEDIANS.get(feat, default))
        if use_fraud:
            # push towards extreme (higher risk) using 99th percentile if known
            if df_stats is not None:
                default = float(pct(df_stats[feat], 99))
            else:
                default = default + (hi - lo) * 0.8

        user_vals[feat] = cols[i % 3].slider(
            label, min_value=float(lo), max_value=float(hi), value=float(default)
        )

    submitted = st.form_submit_button("Predict")

# ============ PREDICT ============
if submitted:
    # build full feature vector from medians then overwrite with user inputs
    row_dict = {f: float(MEDIANS.get(f, 0.0)) for f in ALL_FEATURES}
    for f, v in user_vals.items():
        row_dict[f] = float(v)

    row = pd.DataFrame([row_dict])[ALL_FEATURES]
    proba = float(model.predict_proba(row)[:, 1][0])
    pred  = int(proba >= thr)

    # Pretty result
    st.subheader("Result")
    label = "Fraud" if pred == 1 else "Legit"
    msg = f"**Prediction:** {label}  |  **Probability:** {proba:.3f}  |  **Threshold:** {thr:.3f}"
    (st.error if pred == 1 else st.success)(msg, icon="🚨" if pred == 1 else "✅")

    # Visualization: probability vs threshold
    st.markdown("#### Probability vs Threshold")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(["Fraud Probability", "Decision Threshold"], [proba, thr])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

    # Simple explainability: top contributing features (deviation from median)
    st.markdown("#### Top Contributing Features")
    if df_stats is not None:
        # use robust scale by IQR
        contrib = {}
        for feat in BEST_FEATURES:
            vals = df_stats[feat].values
            q1, q3 = np.nanpercentile(vals, 25), np.nanpercentile(vals, 75)
            iqr = max(q3 - q1, 1e-8)
            contrib[feat] = abs(row_dict[feat] - MEDIANS[feat]) / iqr
        expl = (
            pd.Series(contrib)
            .sort_values(ascending=False)
            .rename("contribution (|value - median| / IQR)")
        )
        # show friendly labels
        expl.index = [FRIENDLY_NAMES.get(k, k) for k in expl.index]
        st.dataframe(expl.to_frame().head(6))
    else:
        st.info("Place `creditcard.csv` next to the app to show feature contributions.")

    st.download_button(
        "⬇️ Download this prediction as CSV",
        row.assign(fraud_probability=proba, fraud_pred=pred).to_csv(index=False).encode("utf-8"),
        "single_prediction.csv",
        "text/csv"
    )

# ============ QUICK DEMO (dataset samples) ============
st.markdown("---")
st.markdown("## Quick Demo (use real samples from dataset)")
if Path(DATA_PATH).exists():
    try:
        df_demo = pd.read_csv(DATA_PATH, nrows=60000)
        if "Class" in df_demo.columns:
            # ensure expected columns exist
            for c in ALL_FEATURES:
                if c not in df_demo.columns:
                    df_demo[c] = 0.0
            df_demo = df_demo[ALL_FEATURES + ["Class"]]

            col1, col2 = st.columns(2)

            if col1.button("▶ Sample Legit Transaction"):
                legit = df_demo[df_demo["Class"] == 0].sample(1, random_state=11)
                x = legit.drop(columns=["Class"])
                probaL = float(model.predict_proba(x)[:, 1][0])
                predL  = int(probaL >= thr)
                st.write("**Input (best features):**")
                st.dataframe(x[BEST_FEATURES].rename(index={x.index[0]:"value"}).T)
                (st.success if predL == 0 else st.error)(
                    f"Predicted: {'Legit (0)' if predL == 0 else 'Fraud (1)'} | Probability={probaL:.3f}",
                    icon="✅" if predL == 0 else "🚨"
                )

            if col2.button("▶ Sample Fraud Transaction"):
                fraud_rows = df_demo[df_demo["Class"] == 1]
                if len(fraud_rows) == 0:
                    st.warning("No fraud rows found in the sample.", icon="⚠️")
                else:
                    fraud = fraud_rows.sample(1, random_state=22)
                    x = fraud.drop(columns=["Class"])
                    probaF = float(model.predict_proba(x)[:, 1][0])
                    predF  = int(probaF >= thr)
                    st.write("**Input (best features):**")
                    st.dataframe(x[BEST_FEATURES].rename(index={x.index[0]:"value"}).T)
                    (st.error if predF == 1 else st.warning)(
                        f"Predicted: {'Fraud (1)' if predF == 1 else 'Legit (0)'} | Probability={probaF:.3f}",
                        icon="🚨" if predF == 1 else "⚠️"
                    )
        else:
            st.info("Dataset has no 'Class' column — demo buttons disabled.")
    except Exception as e:
        st.info(f"Could not load demo data: {e}")
else:
    st.info(f"`{DATA_PATH}` not found — place it next to the app to enable demo buttons.")

st.markdown("---")
st.caption("Best-feature input UI • Adjustable threshold • Probability chart • Simple explanations • Quick demo")
