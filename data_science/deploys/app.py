import os
import json
import traceback
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from huggingface_hub import hf_hub_download, list_repo_files

"""
Credit Risk Monitoring Dashboard + Management System.
Single-page Streamlit app with four tabs:
  1. Portfolio Overview      — metrics, distributions, champion scoreboard
  2. Score Applicant         — interactive scoring using the 6 challengers
  3. Model Monitoring        — Gini / KS / PSI (from model_metrics + monitoring views)
  4. Fraud Risk (preview)    — placeholder until fraud model lands
"""

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
REPO_ID = "Mipjeiger/credit-risk-challengers"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

st.set_page_config(
    page_title="Credit Risk — Monitoring Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ------------------------------------------------------------------
# Cached loaders (run once per session)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model bundle…")
def load_models():
    files = list_repo_files(repo_id=REPO_ID, token=HF_TOKEN)
    model_files = sorted(
        f for f in files if f.startswith("models/") and f.endswith(".joblib")
    )
    models = {}
    for f in model_files:
        name = f.split("/")[-1].removesuffix(".joblib").replace("_", " ").title()
        models[name] = joblib.load(
            hf_hub_download(repo_id=REPO_ID, filename=f, token=HF_TOKEN)
        )
    return models

@st.cache_resource(show_spinner="Loading preprocessors…")
def load_preprocessors():
    meta = json.load(open(hf_hub_download(
        repo_id=REPO_ID, filename="metadata/metadata.json", token=HF_TOKEN)))
    scaler = joblib.load(hf_hub_download(
        repo_id=REPO_ID, filename="metadata/scaler.joblib", token=HF_TOKEN))
    encoders = joblib.load(hf_hub_download(
        repo_id=REPO_ID, filename="metadata/label_encoders.joblib", token=HF_TOKEN))
    return meta, scaler, encoders

@st.cache_data(show_spinner="Loading metrics…")
def load_metrics():
    """model_metrics.csv from the challenger repo."""
    path = hf_hub_download(
        repo_id=REPO_ID, filename="metrics/model_metrics.csv", token=HF_TOKEN)
    return pd.read_csv(path)

@st.cache_data(show_spinner="Loading sample data…")
def load_portfolio_sample(n: int = 5000):
    """Optional: pull a sample for the overview tab. Degrade gracefully if missing."""
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename="data/portfolio_sample.parquet",
            token=HF_TOKEN,
        )
        return pd.read_parquet(path).head(n)
    except Exception:
        return None

# Configuration helper
MODELS = load_models()
META, SCALER, LABEL_ENCODERS = load_preprocessors()
FEATURES = META["feature_columns"]
MODEL_CHOICES = sorted(MODELS.keys())
DEFAULT_MODEL = "Xgboost" if "Xgboost" in MODELS else MODEL_CHOICES[0]

# ------------------------------------------------------------------
# Scoring helpers
# ------------------------------------------------------------------
def prepare_input(raw: dict) -> pd.DataFrame:
    """Take a dict of user inputs, encode categoricals, scale, return DataFrame."""
    df = pd.DataFrame([{c: raw.get(c, 0) for c in FEATURES}])
    df = df[FEATURES]

    for col, le in LABEL_ENCODERS.items():
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            )

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return pd.DataFrame(SCALER.transform(df), columns=df.columns)

def score_one(model_name: str, raw: dict) -> dict:
    model = MODELS[model_name]
    X = prepare_input(raw)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = [int(c) for c in model.classes_]
    return {
        "model": model_name,
        "predicted_class": pred,
        "confidence": float(proba.max()),
        "class_probabilities": {f"class_{c}": float(p) for c, p in zip(classes, proba)},
    }

def score_all(raw: dict) -> pd.DataFrame:
    rows = []
    for name in MODEL_CHOICES:
        try:
            r = score_one(name, raw)
            rows.append({
                "model": name,
                "predicted_class": r["predicted_class"],
                "confidence": round(r["confidence"], 4),
                "p_class_0": round(r["class_probabilities"].get("class_0", np.nan), 4),
                "p_class_1": round(r["class_probabilities"].get("class_1", np.nan), 4),
                "p_class_2": round(r["class_probabilities"].get("class_2", np.nan), 4),
                "p_class_3": round(r["class_probabilities"].get("class_3", np.nan), 4),
            })
        except Exception as e:
            rows.append({"model": name, "predicted_class": None,
                         "confidence": None, "error": str(e)})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# LLM explanation (HF router)
# ------------------------------------------------------------------
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HF_LLM_MODEL  = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

SYSTEM_PROMPT = (
    "You are a senior credit-risk analyst. Explain model predictions in plain "
    "business language for a credit manager. Be concise (max 6 sentences). "
    "State: (1) the decision and what it means, (2) the top 2-3 drivers from the "
    "feature values, (3) the recommended next action. Do not invent numbers that "
    "are not in the input."
)

def explain_prediction(result: dict, raw: dict) -> str:
    """Call the HF router for a grounded explanation of the model prediction."""
    if not HF_TOKEN:
        return "⚠️ HF_TOKEN not set. Cannot call LLM for explanation."

    proba_lines = "\n".join(
        f" - {k}: {v:.2%}" for k, v in result["class_probabilities"].items()
    )
    feature_lines = "\n".join(f" - {k}: {v}" for k, v in raw.items())

    # User messages
    user_msg = (
        f"MODEL: {result['model']}\n"
        f"PREDICTED CLASS: {result["predicted_class"]}\n"
        f"CONFIDENCE: {result['confidence']:.2%}\n"
        f"CLASS PROBABILITIES:\n{proba_lines}\n\n"
        f"APPLICANT FEATURES:\n{feature_lines}\n\n"
        "Write the explanation now."
    )

    try:
        r = requests.post(
            HF_ROUTER_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": HF_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "max_tokens": 400,
                "temperature": 0.2,
            },
            timeout=45,
        )
        if not r.ok:
            return f"❌ LLM error {r.status_code}: {r.text[:300]}"

        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.Timeout:
        return "❌ LLM request timed out. Try again."
    except Exception:
        return f"❌ LLM call failed:\n{traceback.format_exc()}"

# ------------------------------------------------------------------
# Tab 1 — Portfolio Overview
# ------------------------------------------------------------------
def tab_overview():
    st.subheader("Portfolio Overview")

    metrics_df = load_metrics()
    sample_df = load_portfolio_sample()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Challenger models", len(MODELS))
    c2.metric("Features", len(FEATURES))
    c3.metric("Best ROC AUC",
              f"{metrics_df['roc_auc'].max():.4f}"
              if "roc_auc" in metrics_df.columns else "—")
    c4.metric("Best F1",
              f"{metrics_df['f1'].max():.4f}"
              if "f1" in metrics_df.columns else "—")

    st.markdown("#### Challenger scoreboard")
    st.dataframe(
        metrics_df.sort_values(
            "roc_auc" if "roc_auc" in metrics_df.columns else metrics_df.columns[0],
            ascending=False,
        ),
        use_container_width=True,
    )

    if sample_df is not None:
        st.markdown("#### Target distribution (sample)")
        target_col = "approved_flag" if "approved_flag" in sample_df.columns else None
        if target_col:
            fig = px.histogram(sample_df, x=target_col,
                               title="Approved_Flag distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Score distribution")
        if "credit_score" in sample_df.columns:
            fig = px.histogram(sample_df, x="credit_score", nbins=40,
                               title="Credit score distribution")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload `data/portfolio_sample.parquet` to the model repo "
                "to see portfolio charts here.")


# ------------------------------------------------------------------
# Tab 2 — Score Applicant
# ------------------------------------------------------------------
def tab_score():
    st.subheader("Score an Applicant")

    model_name = st.selectbox("Model", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_MODEL))

    st.markdown("##### Feature inputs")
    raw = {}
    cols = st.columns(4)
    for i, col in enumerate(FEATURES):
        with cols[i % 4]:
            if col in LABEL_ENCODERS:
                choices = list(LABEL_ENCODERS[col].classes_)
                raw[col] = st.selectbox(col, choices, key=f"in_{col}")
            else:
                raw[col] = st.number_input(col, value=0.0, key=f"in_{col}")

    if st.button("Evaluate Credit Risk", type="primary"):
        with st.spinner("Scoring…"):
            try:
                result = score_one(model_name, raw)
                st.success(f"Predicted class: **{result['predicted_class']}** "
                           f"(confidence {result['confidence']:.2%})")

                proba_df = pd.DataFrame(
                    {"class": list(result["class_probabilities"].keys()),
                     "probability": list(result["class_probabilities"].values())})
                fig = px.bar(proba_df, x="class", y="probability",
                             title="Class probabilities")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### All challengers on the same input")
                st.dataframe(score_all(raw), use_container_width=True)
            except Exception:
                st.error("Prediction failed")
                st.code(traceback.format_exc())


# ------------------------------------------------------------------
# Tab 3 — Model Monitoring
# ------------------------------------------------------------------
def tab_monitoring():
    st.subheader("Model Monitoring")

    st.caption(
        "In production these come from `credit_risk.model_monitoring`. "
        "In this demo they're recomputed from the challenger holdout."
    )

    metrics = load_metrics()

    # Compute Gini from roc_auc if present
    if "roc_auc" in metrics.columns:
        metrics["gini"] = 2 * metrics["roc_auc"] - 1
    if "ks" not in metrics.columns:
        metrics["ks"] = np.nan
    if "psi" not in metrics.columns:
        metrics["psi"] = np.nan

    THRESH = {"gini": 0.30, "ks": 0.20, "psi": 0.25}

    fig = go.Figure()
    fig.add_bar(x=metrics["model"], y=metrics["gini"], name="Gini",
                marker_color="#4C78A8")
    fig.add_hline(y=THRESH["gini"], line_dash="dash", line_color="red",
                  annotation_text="Gini floor")
    fig.update_layout(title="Gini by model", xaxis_title="", yaxis_title="Gini",
                      height=380)
    st.plotly_chart(fig, use_container_width=True)

    alerts = metrics[
        (metrics["gini"] < THRESH["gini"]) |
        (metrics["ks"].fillna(1) < THRESH["ks"]) |
        (metrics["psi"].fillna(0) > THRESH["psi"])
    ]
    if alerts.empty:
        st.success("All models within thresholds ✅")
    else:
        st.warning(f"{len(alerts)} model(s) breach a threshold")
        st.dataframe(alerts, use_container_width=True)


# ------------------------------------------------------------------
# Tab 4 — Fraud Risk (preview)
# ------------------------------------------------------------------
def tab_fraud():
    st.subheader("Fraud Risk (preview)")
    st.info(
        "Fraud model is under development in `Notebooks/`. "
        "This tab will host:\n\n"
        "- **Fraud probability** per applicant (XGBoost/LightGBM)\n"
        "- **Fraud typology** (from your `Fraud_Typologies_and_Red Flags.pdf`)\n"
        "- **Red-flag list** with rule IDs and policy citations\n"
        "- **Alert queue** ranked by expected loss × probability"
    )

    st.markdown("#### Planned inputs")
    st.code(
        "incoming application features\n"
        "  ├─ device / IP signals\n"
        "  ├─ velocity counters (last 1h / 24h / 7d)\n"
        "  ├─ beneficiary / merchant identity\n"
        "  └─ cross-border / channel flags"
    )

    st.markdown("#### Planned outputs")
    st.code(
        "{\n"
        '  "fraud_probability": 0.0-1.0,\n'
        '  "typology": "account_takeover | first_party | synthetic_id | ...",\n'
        '  "red_flags": ["RF_001", "RF_017"],\n'
        '  "policy_refs": ["Fraud_Typologies_and_Red Flags.pdf p12"],\n'
        '  "recommended_action": "review | block | step-up_auth",\n'
        '  "expected_loss": 1234.56\n'
        "}"
    )

# ------------------------------------------------------------------
# Tab 5 — LLM Explain
# ------------------------------------------------------------------
def tab_explain():
    st.subheader("LLM Explanation")
    st.caption(
        "Uses the Hugging Face router to generate a grounded, "
        "business-language explanation of a single prediction."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        model_name = st.selectbox(
            "Model", MODEL_CHOICES,
            index=MODEL_CHOICES.index(DEFAULT_MODEL),
            key="explain_model",
        )
        st.markdown("##### Inputs")
        raw = {}
        for col in FEATURES:
            if col in LABEL_ENCODERS:
                raw[col] = st.selectbox(
                    col, list(LABEL_ENCODERS[col].classes_),
                    key=f"exp_{col}",
                )
            else:
                raw[col] = st.number_input(col, value=0.0, key=f"exp_{col}")

        run_btn = st.button("Explain", type="primary", key="explain_run")

    with col2:
        if run_btn:
            with st.spinner("Scoring and generating explanation…"):
                try:
                    result = score_one(model_name, raw)

                    st.markdown(
                        f"**Prediction:** class `{result['predicted_class']}` "
                        f"· confidence `{result['confidence']:.2%}`"
                    )

                    proba_df = pd.DataFrame({
                        "class": list(result["class_probabilities"].keys()),
                        "probability": list(result["class_probabilities"].values()),
                    })
                    st.plotly_chart(
                        px.bar(proba_df, x="class", y="probability",
                               title="Class probabilities"),
                        use_container_width=True,
                    )

                    st.markdown("##### LLM explanation")
                    explanation = explain_prediction(result, raw)
                    st.markdown(explanation)

                except Exception:
                    st.error("Pipeline failed")
                    st.code(traceback.format_exc())
        else:
            st.info("Set inputs on the left and click **Explain**.")

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
def main():
    st.title("🏦 Credit Risk — Monitoring Dashboard")
    st.caption(
        f"Challenger repo: `{REPO_ID}` · "
        f"{len(MODELS)} models · {len(FEATURES)} features"
    )

    tabs = st.tabs([
        "📊 Overview",
        "🧮 Score Applicant",
        "📈 Monitoring",
        "🛡️ Fraud (preview)",
        "💬 LLM Explain",
    ])
    with tabs[0]: tab_overview()
    with tabs[1]: tab_score()
    with tabs[2]: tab_monitoring()
    with tabs[3]: tab_fraud()
    with tabs[4]: tab_explain()


if __name__ == "__main__":
    main()