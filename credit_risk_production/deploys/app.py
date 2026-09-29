import os
import json
import traceback
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from dotenv import load_dotenv
from deploys.api_client import predict as api_predict

# ============================================================
# CONFIG
# ============================================================
BASE_PATH = Path(__file__).parent
ENV_PATH = BASE_PATH / ".env"
load_dotenv(ENV_PATH)

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

# Helper Directory Paths
APP_DIR   = Path(__file__).resolve().parent              # .../data_science/deploys
DS_DIR    = APP_DIR.parent                               # .../data_science
REPO_ROOT = DS_DIR.parent                                # .../credit_risk

# Credit-risk artifacts
CR_DIR         = REPO_ROOT / "credit_risk_production" / "models" / "credit_risk"
CR_MODEL_DIR   = CR_DIR / "ml_credit_risk"
CR_META_FILE   = CR_DIR / "metadata_credit_risk" / "metadata.json"
CR_METRIC_FILE = CR_DIR / "metrics_credit_risk" / "model_metrics.csv"
CR_BUNDLE_FILE = (REPO_ROOT/ "credit_risk_production"/ "database" / "LLM" / "outputs_llm" / "model_artifacts" / "model_bundle.joblib")

# Fraud artifacts (used later, wired up now for completeness)
FRAUD_DIR   = DS_DIR / "models" / "fraud_models"
FRAUD_META  = DS_DIR / "models" / "metadata" / "metadata.json"
FRAUD_METRIC = DS_DIR / "models" / "metrics" / "model_metrics.csv"
FRAUD_BUNDLE = FRAUD_DIR / "model_bundle.joblib"

# ============================================================
# CUSTOM REACT-LIKE STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Credit Risk | Management System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ---------- Global ---------- */
[data-testid="stAppViewContainer"] {
    background: #f7f8fc;
}

[data-testid="stHeader"] {
    background: transparent;
}

.main .block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2937;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 0.55rem 0.75rem;
    border-radius: 10px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #1f2937;
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* ---------- Typography ---------- */
.dashboard-title {
    font-size: 2rem;
    font-weight: 750;
    color: #111827;
    letter-spacing: -0.03em;
    margin-bottom: 0.1rem;
}

.dashboard-subtitle {
    color: #6b7280;
    font-size: 0.92rem;
    margin-bottom: 1.3rem;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
    margin-top: 1.5rem;
    margin-bottom: 0.7rem;
}

/* ---------- Header ---------- */
.header-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 650;
}

/* ---------- KPI cards ---------- */
.kpi-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    min-height: 115px;
}

.kpi-label {
    color: #6b7280;
    font-size: 0.78rem;
    font-weight: 600;
}

.kpi-value {
    color: #111827;
    font-size: 1.75rem;
    font-weight: 750;
    margin-top: 0.3rem;
}

.kpi-meta {
    color: #9ca3af;
    font-size: 0.75rem;
    margin-top: 0.25rem;
}

/* ---------- Cards ---------- */
.ui-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 1.15rem;
    margin-bottom: 1rem;
}

.card-title {
    color: #111827;
    font-weight: 700;
    font-size: 0.98rem;
}

.card-subtitle {
    color: #6b7280;
    font-size: 0.78rem;
    margin-top: 0.2rem;
}

/* ---------- Model cards ---------- */
.model-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 1rem;
    min-height: 150px;
}

.model-name {
    color: #111827;
    font-weight: 700;
}

.model-score {
    color: #111827;
    font-size: 1.4rem;
    font-weight: 750;
    margin-top: 0.45rem;
}

.progress-bg {
    height: 7px;
    width: 100%;
    background: #e5e7eb;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.7rem;
}

.progress-fill {
    height: 100%;
    background: #111827;
    border-radius: 999px;
}

/* ---------- Decision cards ---------- */
.decision-approved {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 15px;
    padding: 1rem;
}

.decision-review {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 15px;
    padding: 1rem;
}

.decision-title {
    font-size: 1.2rem;
    font-weight: 750;
}

/* ---------- Alert ---------- */
.alert-card {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 14px;
    padding: 0.9rem 1rem;
}

.success-card {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 14px;
    padding: 0.9rem 1rem;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #d1d5db;
    font-weight: 650;
    min-height: 42px;
}

.stButton > button[kind="primary"] {
    background: #111827;
    border-color: #111827;
}

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div,
.stNumberInput input {
    border-radius: 9px !important;
}

/* ---------- Tables ---------- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .dashboard-title {
        font-size: 1.55rem;
    }

    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)
# ============================================================
# HELPERS
# ============================================================

def card(title, subtitle="", body=""):
    st.markdown(
        f"""
        <div class="ui-card">
            <div class="card-title">{title}</div>
            {"<div class='card-subtitle'>" + subtitle + "</div>" if subtitle else ""}
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, meta=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def model_card(model_name, score, metric_name="ROC AUC"):
    pct = max(0, min(100, float(score) * 100))
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-name">{model_name}</div>
            <div class="model-score">{score:.4f}</div>
            <div class="card-subtitle">{metric_name}</div>
            <div class="progress-bg">
                <div class="progress-fill" style="width:{pct:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# METADATA
# ============================================================

# Credit risk features list
with open(CR_META_FILE) as f:
    FEATURES = json.load(f)["feature_columns"]

# Fraud features list
FRAUD_FEATURES = []
if FRAUD_META.exists():
    with open(FRAUD_META) as f:
        FRAUD_FEATURES = json.load(f).get("feature_columns", [])

# Label encoders for building categorical
LABEL_ENCODERS = {}
if CR_BUNDLE_FILE.exists():
    _cr_bundle = joblib.load(CR_BUNDLE_FILE)
    LABEL_ENCODERS = _cr_bundle.get("label_encoders", {})

FRAUD_LABEL_ENCODERS = {}
if FRAUD_BUNDLE.exists():
    _fraud_bundle = joblib.load(FRAUD_BUNDLE)
    FRAUD_LABEL_ENCODERS = _fraud_bundle.get("label_encoders", {})

# ============================================================
# LOADERS DEPENDENCIES COMPONENTS
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_choices():
    try:
        r = requests.get(f"{API_BASE_URL}/models")
        r.raise_for_status()
        return sorted(r.json()["models"])
    except Exception:
        # Fallback: hardcode to load models
        return ["Logistic Regression", "Random Forest", "Gradient Boosting",
                "XGBoost", "K-Nearest Neighbors", "Decision Tree"]

@st.cache_data(ttl=60, show_spinner=False)
def fetch_fraud_model_choices():
    try:
        r = requests.get(f"{API_BASE_URL}/fraud/models", timeout=10)
        r.raise_for_status()
        return sorted(r.json()["models"])
    except Exception:
        # Fallback: hardcode to load models
        return ["Logistic Regression", "Random Forest", "Gradient Boosting",
                "XGBoost", "K-Nearest Neighbors", "Decision Tree"]

@st.cache_data(ttl=60, show_spinner="Loading model metrics...")
def load_metrics():
    if not CR_METRIC_FILE.exists():
        raise FileNotFoundError(f"Metrics file not found: {CR_METRIC_FILE}")
    df = pd.read_csv(CR_METRIC_FILE)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]
    df = df.rename(columns={
        "model_name": "model", "modelname": "model",
        "f1_weighted": "f1", "f1_score": "f1"
    })

    if "model" not in df.columns:
        st.error("❌ Metrics file does not contain 'model' column.")
        st.stop()
    return df

@st.cache_data(ttl=60, show_spinner="Loading fraud metrics...")
def load_fraud_metrics():
    if not FRAUD_METRIC.exists():
        return None
    df = pd.read_csv(FRAUD_METRIC)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]
    df = df.rename(columns={
        "model_name": "model", "modelname": "model",
        "f1_weighted": "f1", "f1_score": "f1",
    })
    return df

def load_portfolio_sample(n=5000):
    for path in [DS_DIR / "data" / "portfolio_sample.parquet", REPO_ROOT / "credit_risk_production" / "database" / "data" / "portfolio_sample.parquet"]:
        if path.exists():
            return pd.read_parquet(path).head(n)

    return None # If no sample file is found, return None

# ===========================================================
# RETRIEVING LOADER FUNCTIONS
# ===========================================================
# Insert metdata by retrieving
MODEL_CHOICES = fetch_model_choices()
DEFAULT_MODEL= "Gradient Boosting" if "Gradient Boosting" in MODEL_CHOICES else MODEL_CHOICES[0]

FRAUD_MODEL_CHOICES = fetch_fraud_model_choices()
FRAUD_DEFAULT_MODEL = "XGBoost" if "XGBoost" in FRAUD_MODEL_CHOICES else FRAUD_MODEL_CHOICES[0]

# ============================================================
# SCORING
# ============================================================
def score_one(model_name, raw):
    """Delegate to the FastAPI /predict endpoint"""
    try:
        result = api_predict(raw, model_name=model_name)
        return {
            "model": result.get("model_used", model_name),
            "predicted_class": int(result["predicted_flag"]),
            "confidence": float(result["primary_risk_probability"]),
            "class_probabilities": result["class_probabilities"]
        }
    except requests.HTTPError as e:
        body = e.response.text[:500] if e.response is not None else ""
        code = e.response.status_code if e.response is not None else "N/A"
        raise RuntimeError(f"API /predict failed: {e}\n{body}") from e

    except requests.RequestException as e:
        raise RuntimeError(f"API /predict request failed: {e}") from e

def fraud_score_one(model_name, raw):
    """Delegate to the FastAPI /fraud/predict endpoint"""
    try:
        r = requests.post(
            f"{API_BASE_URL}/fraud/predict",
            params={"model_name": model_name} if model_name else {},
            json={"features": raw},
            timeout=30
        )
        r.raise_for_status()
        result = r.json()
        return {
            "model": result.get("model_used", model_name),
            "predicted_class": int(result["predicted_flag"]),
            "confidence": float(result.get("primary_fraud_probability", 0.0)),
            "class_probabilities": result.get("class_probabilities", {})
        }
    except requests.HTTPError as e:
        body = e.response.text[:500] if e.response is not None else ""
        code = e.response.status_code if e.response is not None else "?"
        raise RuntimeError(f"API {code}: {body}") from e
    except requests.RequestException as e:
        raise RuntimeError(f"API unreachable: {e}") from e

def score_all(raw):
    rows = []

    for name in MODEL_CHOICES:
        try:
            result = score_one(
                name,
                raw,
            )

            rows.append(
                {
                    "model": name,
                    "predicted_class": result[
                        "predicted_class"
                    ],
                    "confidence": round(
                        result["confidence"],
                        4,
                    ),
                    "p_class_0": round(
                        result[
                            "class_probabilities"
                        ].get(
                            "class_0",
                            np.nan,
                        ),
                        4,
                    ),
                    "p_class_1": round(
                        result[
                            "class_probabilities"
                        ].get(
                            "class_1",
                            np.nan,
                        ),
                        4,
                    ),
                    "p_class_2": round(
                        result[
                            "class_probabilities"
                        ].get(
                            "class_2",
                            np.nan,
                        ),
                        4,
                    ),
                    "p_class_3": round(
                        result[
                            "class_probabilities"
                        ].get(
                            "class_3",  
                            np.nan,
                        ),
                        4,
                    ),
                }
            )

        except Exception as e:
            rows.append(
                {
                    "model": name,
                    "predicted_class": None,
                    "confidence": None,
                    "error": str(e),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# LLM EXPLANATION
# ============================================================

SYSTEM_PROMPT = (
    "You are a senior credit-risk analyst. "
    "Explain model predictions in plain business language "
    "for a credit manager. Be concise (max 6 sentences). "
    "State: (1) the decision and what it means, "
    "(2) the top 2-3 drivers from the feature values, "
    "(3) the recommended next action. "
    "Do not invent numbers that are not in the input."
)


def explain_prediction(result, raw):
    if not HF_TOKEN:
        return (
            "⚠️ HUGGINGFACE_API_KEY is not set. "
            "Cannot call the LLM."
        )

    proba_lines = "\n".join(
        f" - {k}: {v:.2%}"
        for k, v in result[
            "class_probabilities"
        ].items()
    )

    feature_lines = "\n".join(
        f" - {k}: {v}"
        for k, v in raw.items()
    )

    user_msg = (
        f"MODEL: {result['model']}\n"
        f"PREDICTED CLASS: {result['predicted_class']}\n"
        f"CONFIDENCE: {result['confidence']:.2%}\n"
        f"CLASS PROBABILITIES:\n{proba_lines}\n\n"
        f"APPLICANT FEATURES:\n{feature_lines}\n\n"
        "Write the explanation now."
    )

    try:
        response = requests.post(
            HF_ROUTER_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": HF_LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_msg,
                    },
                ],
                "max_tokens": 400,
                "temperature": 0.2,
            },
            timeout=45,
        )

        if not response.ok:
            return (
                f"❌ LLM error "
                f"{response.status_code}: "
                f"{response.text[:300]}"
            )

        data = response.json()

        return (
            data["choices"][0]["message"]["content"]
            .strip()
        )

    except requests.Timeout:
        return (
            "❌ LLM request timed out. "
            "Please try again."
        )

    except Exception:
        return (
            "❌ LLM call failed:\n\n"
            + traceback.format_exc()
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    try:
        from deploys.api_client import health as api_health
        h = api_health()
        st.markdown('<span class="status-badge">● API healthy</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-badge" style="background:#fee2e2;color:#991b1b;">● API down</span>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="font-size:1.25rem;font-weight:750;margin-bottom:0.2rem;">
            🏦 Credit Risk
        </div>
        <div style="color:#9ca3af;font-size:0.8rem;margin-bottom:1.5rem;">
            Management System
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        f"""
        <div style="font-size:0.75rem;color:#9ca3af;">
            MODEL REPOSITORY
        </div>
        <div style="
            font-size:0.72rem;
            margin-top:0.35rem;
            word-break:break-word;
            color:#e5e7eb;
        ">
            {CR_MODEL_DIR}
        </div>

        <div style="margin-top:1rem;font-size:0.75rem;color:#9ca3af;">
            MODELS
        </div>
        <div style="font-size:0.9rem;font-weight:700;margin-top:0.25rem;">
            {len(MODEL_CHOICES)} challengers
        </div>

        <div style="margin-top:1rem;font-size:0.75rem;color:#9ca3af;">
            FEATURES
        </div>
        <div style="font-size:0.9rem;font-weight:700;margin-top:0.25rem;">
            {len(FEATURES)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🧮 Score Applicant",
            "📈 Model Monitoring",
            "🛡️ Fraud Risk",
            "💬 LLM Explain",
        ],
        label_visibility="collapsed",
        key="main_navigation",
    )

    st.markdown("---")

    st.markdown(
        f"""
        <div style="font-size:0.75rem;color:#9ca3af;">
            MODEL DIRECTORY
        </div>
        <div style="
            font-size:0.72rem;
            margin-top:0.35rem;
            word-break:break-word;
            color:#e5e7eb;
        ">
            {CR_MODEL_DIR}
        </div>

        <div style="margin-top:1rem;font-size:0.75rem;color:#9ca3af;">
            MODELS
        </div>
        <div style="font-size:0.9rem;font-weight:700;margin-top:0.25rem;">
            {len(MODEL_CHOICES)} challengers
        </div>

        <div style="margin-top:1rem;font-size:0.75rem;color:#9ca3af;">
            FEATURES
        </div>
        <div style="font-size:0.9rem;font-weight:700;margin-top:0.25rem;">
            {len(FEATURES)}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="header-card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;">
            <div>
                <div class="dashboard-title">Credit Risk Monitoring</div>
                <div class="dashboard-subtitle">
                    Merchant onboarding, model scoring, monitoring and LLM-assisted explanations
                </div>
            </div>
            <div class="status-badge">● Production</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

def tab_overview():
    metrics_df = load_metrics()
    sample_df = load_portfolio_sample()

    st.markdown(
        '<div class="section-title">Portfolio Overview</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Challenger models",
            len(MODEL_CHOICES),
            "Active model bundle",
        )

    with c2:
        metric_card(
            "Features",
            len(FEATURES),
            "Production feature set",
        )

    with c3:
        best_auc = (
            metrics_df["roc_auc"].max()
            if "roc_auc" in metrics_df.columns
            else np.nan
        )

        metric_card(
            "Best ROC AUC",
            f"{best_auc:.4f}"
            if pd.notna(best_auc)
            else "—",
            "Model discrimination",
        )

    with c4:
        best_f1 = (
            metrics_df["f1"].max()
            if "f1" in metrics_df.columns
            else np.nan
        )

        metric_card(
            "Best F1",
            f"{best_f1:.4f}"
            if pd.notna(best_f1)
            else "—",
            "Classification balance",
        )

    st.markdown(
        '<div class="section-title">Challenger Models</div>',
        unsafe_allow_html=True,
    )

    if "roc_auc" in metrics_df.columns:
        leaderboard = (
            metrics_df
            .sort_values(
                "roc_auc",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        model_columns = st.columns(
            min(3, len(leaderboard))
        )

        for i, row in leaderboard.iterrows():
            with model_columns[
                i % len(model_columns)
            ]:
                model_card(
                    str(row["model"]),
                    float(row["roc_auc"]),
                )

    st.markdown(
        '<div class="section-title">Performance Comparison</div>',
        unsafe_allow_html=True,
    )

    if "roc_auc" in metrics_df.columns:
        fig = px.bar(
            leaderboard,
            x="model",
            y="roc_auc",
            text="roc_auc",
            title="ROC AUC by challenger",
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )

        fig.update_layout(
            height=400,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    if sample_df is not None:
        col1, col2 = st.columns(2)

        target_col = (
            "approved_flag"
            if "approved_flag" in sample_df.columns
            else None
        )

        with col1:
            if target_col:
                fig = px.histogram(
                    sample_df,
                    x=target_col,
                    title="Approved Flag Distribution",
                )

                fig.update_layout(
                    height=360,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        with col2:
            if "credit_score" in sample_df.columns:
                fig = px.histogram(
                    sample_df,
                    x="credit_score",
                    nbins=40,
                    title="Credit Score Distribution",
                )

                fig.update_layout(
                    height=360,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

    else:
        st.info(
            "Add data/portfolio_sample.parquet to the "
            "Hugging Face model repository to enable "
            "portfolio distribution charts."
        )

# ============================================================
# APPLICANT INPUT COMPONENT
# ============================================================

def applicant_input(prefix):
    raw = {}

    input_columns = st.columns(3)

    for i, col in enumerate(FEATURES):
        with input_columns[i % 3]:
            key = f"{prefix}_{col}"

            if col in LABEL_ENCODERS:
                choices = list(
                    LABEL_ENCODERS[col].classes_
                )

                raw[col] = st.selectbox(
                    col,
                    choices,
                    key=key,
                )

            else:
                raw[col] = st.number_input(
                    col,
                    value=0.0,
                    key=key,
                )

    return raw


# ============================================================
# TAB 2 — SCORE APPLICANT
# ============================================================

def tab_score():
    st.markdown(
        '<div class="section-title">Score Applicant</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.5, 1],
        gap="large",
    )

    with left:
        card(
            "Applicant Profile",
            "Enter applicant-level features used by the production pipeline.",
        )

        model_name = st.selectbox(
            "Scoring model",
            MODEL_CHOICES,
            index=MODEL_CHOICES.index(
                DEFAULT_MODEL
            ),
            key="score_model",
        )

        raw = applicant_input("score")

        evaluate = st.button(
            "Evaluate Credit Risk",
            type="primary",
            use_container_width=True,
        )

    with right:
        st.markdown(
            """
            <div class="ui-card">
                <div class="card-title">
                    Scoring Workflow
                </div>

                <div class="card-subtitle">
                    Production inference path
                </div>

                <div style="
                    margin-top:1rem;
                    line-height:2;
                    font-size:0.88rem;
                ">
                    <div>① Applicant features</div>
                    <div>↓</div>
                    <div>② Label encoding</div>
                    <div>↓</div>
                    <div>③ Feature scaling</div>
                    <div>↓</div>
                    <div>④ Challenger model</div>
                    <div>↓</div>
                    <div>⑤ Probability + class</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if evaluate:
        with st.spinner("Scoring applicant..."):
            try:
                result = score_one(
                    model_name,
                    raw,
                )

                predicted = result[
                    "predicted_class"
                ]

                confidence = result[
                    "confidence"
                ]

                if predicted == 0:
                    st.markdown(
                        f"""
                        <div class="decision-approved">
                            <div class="decision-title">
                                ✓ Prediction: Class {predicted}
                            </div>
                            <div style="
                                margin-top:0.3rem;
                                color:#047857;
                            ">
                                Model confidence:
                                <b>{confidence:.2%}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="decision-review">
                            <div class="decision-title">
                                ! Prediction: Class {predicted}
                            </div>
                            <div style="
                                margin-top:0.3rem;
                                color:#9a3412;
                            ">
                                Model confidence:
                                <b>{confidence:.2%}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                col1, col2 = st.columns(
                    [1, 1]
                )

                with col1:
                    proba_df = pd.DataFrame(
                        {
                            "class": list(
                                result[
                                    "class_probabilities"
                                ].keys()
                            ),
                            "probability": list(
                                result[
                                    "class_probabilities"
                                ].values()
                            ),
                        }
                    )

                    fig = px.bar(
                        proba_df,
                        x="class",
                        y="probability",
                        title="Class Probabilities",
                    )

                    fig.update_layout(
                        yaxis_tickformat=".0%",
                        height=350,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                with col2:
                    st.markdown(
                        '<div class="section-title">'
                        'Inference Result'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.json(
                        {
                            "model": result["model"],
                            "predicted_class": result[
                                "predicted_class"
                            ],
                            "confidence": round(
                                result[
                                    "confidence"
                                ],
                                4,
                            ),
                            "class_probabilities":
                                result[
                                    "class_probabilities"
                                ],
                        }
                    )

                st.markdown(
                    '<div class="section-title">'
                    'All Challenger Models'
                    '</div>',
                    unsafe_allow_html=True,
                )

                comparison = score_all(raw)

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception:
                st.error(
                    "Prediction failed."
                )

                st.code(
                    traceback.format_exc()
                )


# ============================================================
# TAB 3 — MONITORING
# ============================================================

def tab_monitoring():
    st.markdown(
        '<div class="section-title">Model Monitoring</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Production monitoring is designed around "
        "Gini, KS and PSI thresholds."
    )

    metrics = load_metrics().copy()

    if "roc_auc" in metrics.columns:
        metrics["gini"] = (
            2 * metrics["roc_auc"] - 1
        )
    else:
        metrics["gini"] = np.nan

    if "ks" not in metrics.columns:
        metrics["ks"] = np.nan

    if "psi" not in metrics.columns:
        metrics["psi"] = np.nan

    thresholds = {
        "gini": 0.30,
        "ks": 0.20,
        "psi": 0.25,
    }

    c1, c2, c3 = st.columns(3)

    with c1:
        max_gini = (
            metrics["gini"].max()
            if metrics["gini"].notna().any()
            else np.nan
        )

        metric_card(
            "Gini",
            f"{max_gini:.3f}"
            if pd.notna(max_gini)
            else "—",
            f"Floor {thresholds['gini']:.2f}",
        )

    with c2:
        max_ks = (
            metrics["ks"].max()
            if metrics["ks"].notna().any()
            else np.nan
        )

        metric_card(
            "KS",
            f"{max_ks:.3f}"
            if pd.notna(max_ks)
            else "—",
            f"Floor {thresholds['ks']:.2f}",
        )

    with c3:
        max_psi = (
            metrics["psi"].max()
            if metrics["psi"].notna().any()
            else np.nan
        )

        metric_card(
            "PSI",
            f"{max_psi:.3f}"
            if pd.notna(max_psi)
            else "—",
            f"Ceiling {thresholds['psi']:.2f}",
        )

    st.markdown(
        '<div class="section-title">'
        'Gini by Model'
        '</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_bar(
        x=metrics["model"],
        y=metrics["gini"],
        name="Gini",
    )

    fig.add_hline(
        y=thresholds["gini"],
        line_dash="dash",
        annotation_text="Gini floor",
    )

    fig.update_layout(
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    alerts = metrics[
        (
            metrics["gini"]
            < thresholds["gini"]
        )
        |
        (
            metrics["ks"].fillna(1)
            < thresholds["ks"]
        )
        |
        (
            metrics["psi"].fillna(0)
            > thresholds["psi"]
        )
    ]

    st.markdown(
        '<div class="section-title">'
        'Monitoring Alerts'
        '</div>',
        unsafe_allow_html=True,
    )

    if alerts.empty:
        st.markdown(
            """
            <div class="success-card">
                ✓ All available model metrics are
                within configured thresholds.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-card">
                ⚠ {len(alerts)}
                model(s) breached at least one
                monitoring threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            alerts,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        '<div class="section-title">'
        'Monitoring Table'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 4 — FRAUD
# ============================================================

def tab_fraud():
    st.markdown(
        '<div class="section-title">Fraud Risk</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ui-card">
            <div class="card-title">
                🛡️ Fraud Risk Preview
            </div>

            <div class="card-subtitle">
                Fraud model integration is currently
                under development.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        card(
            "Fraud Probability",
            "XGBoost / LightGBM",
            "Applicant-level fraud probability.",
        )

    with c2:
        card(
            "Fraud Typology",
            "Policy-grounded",
            "Account takeover, synthetic ID, first-party fraud, etc.",
        )

    with c3:
        card(
            "Red Flags",
            "Rule IDs",
            "Policy citations and operational signals.",
        )

    st.markdown(
        '<div class="section-title">'
        'Planned Fraud Pipeline'
        '</div>',
        unsafe_allow_html=True,
    )

    st.code(
        """
Incoming application
        │
        ├── Device / IP signals
        ├── Velocity counters
        ├── Beneficiary / merchant identity
        └── Cross-border / channel flags
        │
        ▼
Fraud ML model
        │
        ├── fraud_probability
        ├── typology
        ├── red_flags
        ├── policy_refs
        ├── recommended_action
        └── expected_loss
        """,
        language="text",
    )

# ============================================================
# TAB 5 — LLM EXPLAIN
# ============================================================

def tab_explain():
    st.markdown(
        '<div class="section-title">LLM Explanation</div>',
        unsafe_allow_html=True,
    )

    # Initialize chat history for this session
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Credit Risk Assistant. How can I help you today?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about credit risk..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Construct messages for the API using st.session_state.messages
                api_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                api_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

                response = requests.post(
                    HF_ROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {HF_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": HF_LLM_MODEL,
                        "messages": api_messages,
                        "max_tokens": 512,
                        "temperature": 0.3
                    },
                    timeout=45,
                )
                response.raise_for_status()
                data = response.json()
                full_response = data["choices"][0]["message"]["content"].strip()

            except Exception as e:
                full_response = f"❌ LLM call failed: {str(e)}"

            # Display the response
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.caption(
        "The LLM explains the ML result; it does not replace "
        "the underlying credit-risk model."
    )

    left, right = st.columns(
        [1.3, 1],
        gap="large",
    )

    with left:
        model_name = st.selectbox(
            "Model",
            MODEL_CHOICES,
            index=MODEL_CHOICES.index(
                DEFAULT_MODEL
            ),
            key="explain_model",
        )

        raw = applicant_input("explain")

        run_llm = st.button(
            "Generate Explanation",
            type="primary",
            use_container_width=True,
            key="explain_run",
        )

    with right:
        st.markdown(
            """
            <div class="ui-card">
                <div class="card-title">
                    LLM Architecture
                </div>

                <div class="card-subtitle">
                    Grounded business explanation
                </div>

                <div style="
                    margin-top:1rem;
                    line-height:2;
                    font-size:0.88rem;
                ">
                    <div>① ML prediction</div>
                    <div>↓</div>
                    <div>② Probability</div>
                    <div>↓</div>
                    <div>③ Applicant features</div>
                    <div>↓</div>
                    <div>④ Hugging Face Router</div>
                    <div>↓</div>
                    <div>⑤ Business explanation</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_llm:
        with st.spinner(
            "Scoring and generating explanation..."
        ):
            try:
                result = score_one(
                    model_name,
                    raw,
                )

                st.markdown(
                    f"""
                    <div class="ui-card">
                        <div class="row">
                            <div>
                                <div class="card-title">
                                    Prediction
                                </div>
                                <div style="
                                    font-size:1.5rem;
                                    font-weight:750;
                                    margin-top:0.2rem;
                                ">
                                    Class
                                    {result['predicted_class']}
                                </div>
                            </div>

                            <div style="
                                text-align:right;
                            ">
                                <div class="card-subtitle">
                                    Confidence
                                </div>
                                <div style="
                                    font-size:1.4rem;
                                    font-weight:750;
                                ">
                                    {result['confidence']:.2%}
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                proba_df = pd.DataFrame(
                    {
                        "class": list(
                            result[
                                "class_probabilities"
                            ].keys()
                        ),
                        "probability": list(
                            result[
                                "class_probabilities"
                            ].values()
                        ),
                    }
                )

                fig = px.bar(
                    proba_df,
                    x="class",
                    y="probability",
                    title="Class Probabilities",
                )

                fig.update_layout(
                    yaxis_tickformat=".0%",
                    height=350,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                explanation = explain_prediction(
                    result,
                    raw,
                )

                st.markdown(
                    """
                    <div class="section-title">
                        Business Explanation
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="ui-card">
                        {explanation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception:
                st.error(
                    "LLM pipeline failed."
                )

                st.code(
                    traceback.format_exc()
                )

    else:
        st.info(
            "Enter applicant features and click "
            "**Generate Explanation**."
        )

# ============================================================
# ROUTER
# ============================================================

if page == "📊 Overview":
    tab_overview()

elif page == "🧮 Score Applicant":
    tab_score()

elif page == "📈 Model Monitoring":
    tab_monitoring()

elif page == "🛡️ Fraud Risk":
    tab_fraud()

elif page == "💬 LLM Explain":
    tab_explain()

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:2rem;
        padding-top:1rem;
        border-top:1px solid #e5e7eb;
        color:#9ca3af;
        font-size:0.72rem;
        text-align:center;
    ">
        Credit Risk Monitoring System ·
        ML Scoring · Model Monitoring · LLM Explanation
    </div>
    """,
    unsafe_allow_html=True,
)
