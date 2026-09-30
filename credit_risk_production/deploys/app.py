import os
import json
import traceback
import time
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from dotenv import load_dotenv
from api_client import predict as api_predict, health as api_health

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

# Helper main root path
_PROJECT_DIR = Path(os.environ.get("CR_PROJECT_DIR", "")).resolve() if os.environ.get("CR_PROJECT_DIR") else None

if _PROJECT_DIR and _PROJECT_DIR.exists():
    PROD_DIR = _PROJECT_DIR
    REPO_ROOT = PROD_DIR.parent
else:
    APP_DIR   = Path(__file__).resolve().parent
    DS_DIR    = APP_DIR.parent
    REPO_ROOT = DS_DIR.parent
    PROD_DIR  = REPO_ROOT / "credit_risk_production"

# Credit-risk artifacts
CR_DIR         = PROD_DIR / "models" / "credit_risk"
CR_META_FILE   = CR_DIR / "metadata_credit_risk" / "metadata.json"
CR_METRIC_FILE = CR_DIR / "metrics_credit_risk" / "model_metrics.csv"
CR_BUNDLE_FILE = PROD_DIR / "database" / "LLM" / "outputs_llm" / "model_artifacts" / "model_bundle.joblib"

# Fraud artifacts
FRAUD_DIR    = PROD_DIR / "models" / "fraud"
FRAUD_META   = FRAUD_DIR / "metadata.json"
FRAUD_METRIC = FRAUD_DIR / "model_metrics.csv"
FRAUD_BUNDLE = FRAUD_DIR / "model_bundle.joblib"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Credit Risk | Management System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One-time font import (cheaper than re-importing every rerun)
st.markdown(
    "<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');</style>",
    unsafe_allow_html=True,
)

# ============================================================
# THEME
# ============================================================
THEMES = {
    "Light": {
        "bg": "#f7f8fc",
        "sidebar_bg": "#0F2A44",
        "sidebar_text": "#E8EEF3",
        "sidebar_hover": "#1a3a5c",
        "card_bg": "#ffffff",
        "card_border": "#E6EBF0",
        "title": "#0F2A44",
        "subtitle": "#6B7C93",
        "text": "#1a3a5c",
        "muted": "#8A9BB0",
        "accent": "#0E7C7B",
        "primary_btn": "#0F2A44",
        "success_bg": "#ecfdf5",
        "success_border": "#a7f3d0",
        "success_text": "#047857",
        "warn_bg": "#fff7ed",
        "warn_border": "#fed7aa",
        "warn_text": "#9a3412",
        "danger_bg": "#fee2e2",
        "danger_border": "#fecaca",
        "danger_text": "#991b1b",
        "plot_bg": "#ffffff",
        "header_grad_start": "#0F2A44",
        "header_grad_end": "#1a3a5c",
    },
    "Dark": {
        "bg": "#0b1220",
        "sidebar_bg": "#111827",
        "sidebar_text": "#e5e7eb",
        "sidebar_hover": "#1f2937",
        "card_bg": "#1f2937",
        "card_border": "#374151",
        "title": "#f3f4f6",
        "subtitle": "#9ca3af",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "accent": "#34d399",
        "primary_btn": "#374151",
        "success_bg": "#064e3b",
        "success_border": "#065f46",
        "success_text": "#a7f3d0",
        "warn_bg": "#78350f",
        "warn_border": "#92400e",
        "warn_text": "#fde68a",
        "danger_bg": "#7f1d1d",
        "danger_border": "#991b1b",
        "danger_text": "#fecaca",
        "plot_bg": "#1f2937",
        "header_grad_start": "#111827",
        "header_grad_end": "#1f2937",
    },
}


def inject_theme():
    """Inject all CSS in ONE place, driven by the active theme."""
    theme_name = st.session_state.get("theme_mode", "Light")
    t = THEMES[theme_name]

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {t['bg']};
            --sidebar-bg: {t['sidebar_bg']};
            --sidebar-text: {t['sidebar_text']};
            --card-bg: {t['card_bg']};
            --card-border: {t['card_border']};
            --title: {t['title']};
            --subtitle: {t['subtitle']};
            --text: {t['text']};
            --muted: {t['muted']};
            --accent: {t['accent']};
            --primary-btn: {t['primary_btn']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: var(--text);
        }}

        .stApp {{ background: var(--bg); }}
        .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }}

        /* -------- Sidebar -------- */
        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        [data-testid="stSidebar"] * {{ color: var(--sidebar-text) !important; }}
        [data-testid="stSidebar"] .stRadio label {{
            border-radius: 8px;
            padding: 6px 10px;
            transition: background 0.15s ease;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{
            background: {t['sidebar_hover']};
        }}

        /* -------- Typography -------- */
        h1, h2, h3 {{ color: var(--title); letter-spacing: -0.01em; }}
        h2 {{
            font-size: 1.25rem;
            margin-top: 1.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.4rem;
        }}

        /* -------- Cards (all variants) -------- */
        .ui-card, .kpi-card, .model-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(15,42,68,0.05);
            margin-bottom: 12px;
        }}
        .card-title, .kpi-label, .model-name {{
            font-weight: 700;
            color: var(--title);
            font-size: 1rem;
        }}
        .card-subtitle, .kpi-meta {{
            color: var(--subtitle);
            font-size: 0.82rem;
            margin-top: 2px;
        }}
        .kpi-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }}
        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 750;
            color: var(--title);
            margin-top: 4px;
        }}

        /* -------- Section titles -------- */
        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--title);
            margin: 1.4rem 0 0.6rem 0;
        }}

        /* -------- Dashboard header -------- */
        .header-card {{
            background: linear-gradient(135deg, {t['header_grad_start']} 0%, {t['header_grad_end']} 100%);
            color: #fff;
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 14px rgba(15,42,68,0.15);
        }}
        .dashboard-title {{
            font-size: 1.6rem;
            font-weight: 750;
            color: #fff;
            letter-spacing: -0.02em;
        }}
        .dashboard-subtitle {{
            color: rgba(255,255,255,0.75);
            font-size: 0.9rem;
            margin-top: 4px;
        }}
        .header-card .status-badge {{
            background: rgba(255,255,255,0.15);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.25);
        }}

        /* -------- Model cards -------- */
        .model-score {{
            font-size: 1.8rem;
            font-weight: 750;
            color: var(--title);
            margin-top: 4px;
        }}
        .progress-bg {{
            background: var(--card-border);
            height: 6px;
            border-radius: 999px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent), {t['accent']});
            border-radius: 999px;
            transition: width 0.4s ease;
        }}

        /* -------- Decision banners -------- */
        .decision-approved, .success-card {{
            background: {t['success_bg']};
            border: 1px solid {t['success_border']};
            border-radius: 12px;
            padding: 14px 18px;
            margin: 8px 0;
            color: {t['success_text']};
        }}
        .decision-review, .alert-card {{
            background: {t['warn_bg']};
            border: 1px solid {t['warn_border']};
            border-radius: 12px;
            padding: 14px 18px;
            margin: 8px 0;
            color: {t['warn_text']};
        }}
        .decision-title {{
            font-weight: 700;
            font-size: 1.05rem;
        }}

        /* -------- Status badge -------- */
        .status-badge {{
            background: {t['success_bg']};
            color: {t['success_text']};
            border: 1px solid {t['success_border']};
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }}

        /* -------- Buttons -------- */
        .stButton > button[kind="primary"] {{
            background: var(--primary-btn);
            border: 1px solid var(--primary-btn);
            border-radius: 10px;
            font-weight: 600;
            transition: transform 0.1s ease, opacity 0.15s ease;
        }}
        .stButton > button[kind="primary"]:hover {{
            opacity: 0.9;
            transform: translateY(-1px);
        }}

        /* -------- Insights / pills -------- */
        .insight-box {{
            background: {t['success_bg']};
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 10px 14px;
            margin: 6px 0;
            font-size: 0.88rem;
            color: var(--text);
        }}
        .insight-warn {{ background: {t['warn_bg']}; border-left-color: {t['warn_border']}; }}
        .insight-bad  {{ background: {t['danger_bg']}; border-left-color: {t['danger_border']}; }}

        .badge-low  {{ background:#E6F4EA; color:#1B6B2E; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; }}
        .badge-med  {{ background:#FFF3CD; color:#7A5A00; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; }}
        .badge-high {{ background:#FCE8E8; color:#9B1C1C; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; }}
        .badge-crit {{ background:#9B1C1C; color:white;    padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; }}

        hr {{ border-color: var(--card-border); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(fig, height=380):
    """Apply theme-aware plotly layout consistently."""
    t = THEMES[st.session_state.get("theme_mode", "Light")]
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor=t["plot_bg"],
        paper_bgcolor=t["plot_bg"],
        font=dict(color=t["text"], family="Inter"),
        title_font=dict(color=t["title"], size=15),
    )
    fig.update_xaxes(gridcolor=t["card_border"], linecolor=t["card_border"])
    fig.update_yaxes(gridcolor=t["card_border"], linecolor=t["card_border"])
    return fig


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
with open(CR_META_FILE) as f:
    FEATURES = json.load(f)["feature_columns"]

FRAUD_FEATURES = []
if FRAUD_META.exists():
    with open(FRAUD_META) as f:
        FRAUD_FEATURES = json.load(f).get("feature_columns", [])

LABEL_ENCODERS = {}
if CR_BUNDLE_FILE.exists():
    _cr_bundle = joblib.load(CR_BUNDLE_FILE)
    LABEL_ENCODERS = _cr_bundle.get("label_encoders", {})

FRAUD_LABEL_ENCODERS = {}
if FRAUD_BUNDLE.exists():
    _fraud_bundle = joblib.load(FRAUD_BUNDLE)
    FRAUD_LABEL_ENCODERS = _fraud_bundle.get("label_encoders", {})


# ============================================================
# LOADERS
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_choices():
    try:
        r = requests.get(f"{API_BASE_URL}/models")
        r.raise_for_status()
        return sorted(r.json()["models"])
    except Exception:
        return ["Logistic Regression", "Random Forest", "Gradient Boosting",
                "XGBoost", "K-Nearest Neighbors", "Decision Tree"]


@st.cache_data(ttl=60, show_spinner=False)
def fetch_fraud_model_choices():
    try:
        r = requests.get(f"{API_BASE_URL}/fraud/models", timeout=10)
        r.raise_for_status()
        return sorted(r.json()["models"])
    except Exception:
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
    for path in [
        DS_DIR / "data" / "portfolio_sample.parquet",
        REPO_ROOT / "credit_risk_production" / "database" / "data" / "portfolio_sample.parquet",
    ]:
        if path.exists():
            return pd.read_parquet(path).head(n)
    return None


# ============================================================
# RETRIEVE MODEL CHOICES
# ============================================================
MODEL_CHOICES = fetch_model_choices()
DEFAULT_MODEL = "Gradient Boosting" if "Gradient Boosting" in MODEL_CHOICES else MODEL_CHOICES[0]

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
            "class_probabilities": result["class_probabilities"],
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
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()
        return {
            "model": result.get("model_used", model_name),
            "predicted_class": int(result["predicted_flag"]),
            "confidence": float(result.get("primary_fraud_probability", 0.0)),
            "class_probabilities": result.get("class_probabilities", {}),
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
            result = score_one(name, raw)
            rows.append({
                "model": name,
                "predicted_class": result["predicted_class"],
                "confidence": round(result["confidence"], 4),
                "p_class_0": round(result["class_probabilities"].get("class_0", np.nan), 4),
                "p_class_1": round(result["class_probabilities"].get("class_1", np.nan), 4),
                "p_class_2": round(result["class_probabilities"].get("class_2", np.nan), 4),
                "p_class_3": round(result["class_probabilities"].get("class_3", np.nan), 4),
            })
        except Exception as e:
            rows.append({
                "model": name,
                "predicted_class": None,
                "confidence": None,
                "error": str(e),
            })
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
        return "⚠️ HUGGINGFACE_API_KEY is not set. Cannot call the LLM."

    proba_lines = "\n".join(
        f" - {k}: {v:.2%}" for k, v in result["class_probabilities"].items()
    )
    feature_lines = "\n".join(f" - {k}: {v}" for k, v in raw.items())

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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 400,
                "temperature": 0.2,
            },
            timeout=45,
        )

        if not response.ok:
            return f"❌ LLM error {response.status_code}: {response.text[:300]}"

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.Timeout:
        return "❌ LLM request timed out. Please try again."
    except Exception:
        return "❌ LLM call failed:\n\n" + traceback.format_exc()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Health status
    try:
        api_health()
        st.markdown('<span class="status-badge">● API healthy</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown(
            '<span class="status-badge" style="background:#fee2e2;color:#991b1b;">● API down</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="font-size:1.25rem;font-weight:750;margin:0.6rem 0 0.2rem;">
            🏦 Credit Risk
        </div>
        <div style="color:#9ca3af;font-size:0.8rem;margin-bottom:1rem;">
            Management System
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- THEME TOGGLE ----
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Light"

    st.radio(
        "Theme",
        ["Light", "Dark"],
        key="theme_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🧮 Score Applicant",
            "📈 Model Monitoring",
            "🛡️ Fraud Risk",
            "💬 LLM Explain",
            "🤖 Chatbot",
        ],
        label_visibility="collapsed",
        key="main_navigation",
    )

    st.markdown("---")

    # ---- Single info block ----
    st.markdown(
        f"""
        <div style="font-size:0.72rem;color:#9ca3af;">MODEL DIRECTORY</div>
        <div style="font-size:0.72rem;margin-top:0.35rem;word-break:break-word;color:#e5e7eb;">
            {CR_DIR}
        </div>
        <div style="margin-top:1rem;font-size:0.72rem;color:#9ca3af;">MODELS</div>
        <div style="font-size:0.9rem;font-weight:700;margin-top:0.25rem;">
            {len(MODEL_CHOICES)} challengers
        </div>
        <div style="margin-top:1rem;font-size:0.72rem;color:#9ca3af;">FEATURES</div>
        <div style="font-size:0.9rem;font-weight:700;margin-top:0.25rem;">
            {len(FEATURES)}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Inject theme AFTER sidebar so theme_mode is set
inject_theme()


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

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Challenger models", len(MODEL_CHOICES), "Active model bundle")

    with c2:
        metric_card("Features", len(FEATURES), "Production feature set")

    with c3:
        best_auc = metrics_df["roc_auc"].max() if "roc_auc" in metrics_df.columns else np.nan
        metric_card(
            "Best ROC AUC",
            f"{best_auc:.4f}" if pd.notna(best_auc) else "—",
            "Model discrimination",
        )

    with c4:
        best_f1 = metrics_df["f1"].max() if "f1" in metrics_df.columns else np.nan
        metric_card(
            "Best F1",
            f"{best_f1:.4f}" if pd.notna(best_f1) else "—",
            "Classification balance",
        )

    # Portfolio distribution (only render if sample exists)
    if sample_df is not None:
        left, right = st.columns(2)

        with left:
            st.markdown('<div class="section-title">Approved Flag Distribution</div>', unsafe_allow_html=True)
            target_col = "approved_flag" if "approved_flag" in sample_df.columns else None
            if target_col:
                fig = px.histogram(sample_df, x=target_col, title="")
                plot_layout(fig, height=340)
                st.plotly_chart(fig, width='stretch', theme=None)

        with right:
            st.markdown('<div class="section-title">Credit Score Distribution</div>', unsafe_allow_html=True)
            if "credit_score" in sample_df.columns:
                fig = px.histogram(sample_df, x="credit_score", nbins=40, title="")
                plot_layout(fig, height=340)
                st.plotly_chart(fig, width='stretch', theme=None)

    st.markdown('<div class="section-title">Challenger Models</div>', unsafe_allow_html=True)

    if "roc_auc" in metrics_df.columns:
        leaderboard = (
            metrics_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
        )

        model_columns = st.columns(min(3, len(leaderboard)))
        for i, row in leaderboard.iterrows():
            with model_columns[i % len(model_columns)]:
                model_card(str(row["model"]), float(row["roc_auc"]))

    st.markdown('<div class="section-title">Performance Comparison</div>', unsafe_allow_html=True)

    if "roc_auc" in metrics_df.columns:
        fig = px.bar(
            leaderboard,
            x="model",
            y="roc_auc",
            text="roc_auc",
            title="",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        plot_layout(fig, height=400)
        st.plotly_chart(fig, width='stretch', theme=None)

    if sample_df is None:
        st.info(
            "Add data/portfolio_sample.parquet to the Hugging Face model repository "
            "to enable portfolio distribution charts."
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
                choices = list(LABEL_ENCODERS[col].classes_)
                raw[col] = st.selectbox(col, choices, key=key)
            else:
                raw[col] = st.number_input(col, value=0.0, key=key)

    return raw


# ============================================================
# TAB 2 — SCORE APPLICANT
# ============================================================
def tab_score():
    st.markdown('<div class="section-title">Score Applicant</div>', unsafe_allow_html=True)

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        card(
            "Applicant Profile",
            "Enter applicant-level features used by the production pipeline.",
        )

        model_name = st.selectbox(
            "Scoring model",
            MODEL_CHOICES,
            index=MODEL_CHOICES.index(DEFAULT_MODEL),
            key="score_model",
        )

        raw = applicant_input("score")

        evaluate = st.button(
            "Evaluate Credit Risk",
            type="primary",
            width='stretch',
        )

    with right:
        st.markdown(
            """
            <div class="ui-card">
                <div class="card-title">Scoring Workflow</div>
                <div class="card-subtitle">Production inference path</div>
                <div style="margin-top:1rem;line-height:2;font-size:0.88rem;">
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
                result = score_one(model_name, raw)
                predicted = result["predicted_class"]
                confidence = result["confidence"]

                if predicted == 0:
                    st.markdown(
                        f"""
                        <div class="decision-approved">
                            <div class="decision-title">✓ Prediction: Class {predicted}</div>
                            <div style="margin-top:0.3rem;">
                                Model confidence: <b>{confidence:.2%}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="decision-review">
                            <div class="decision-title">! Prediction: Class {predicted}</div>
                            <div style="margin-top:0.3rem;">
                                Model confidence: <b>{confidence:.2%}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                col1, col2 = st.columns([1, 1])

                with col1:
                    proba_df = pd.DataFrame({
                        "class": list(result["class_probabilities"].keys()),
                        "probability": list(result["class_probabilities"].values()),
                    })
                    fig = px.bar(proba_df, x="class", y="probability", title="")
                    fig.update_layout(yaxis_tickformat=".0%")
                    plot_layout(fig, height=350)
                    st.plotly_chart(fig, width='stretch', theme=None)

                with col2:
                    st.markdown('<div class="section-title">Inference Result</div>', unsafe_allow_html=True)
                    st.json({
                        "model": result["model"],
                        "predicted_class": result["predicted_class"],
                        "confidence": round(result["confidence"], 4),
                        "class_probabilities": result["class_probabilities"],
                    })

                st.markdown('<div class="section-title">All Challenger Models</div>', unsafe_allow_html=True)
                comparison = score_all(raw)
                st.dataframe(comparison, width='stretch', hide_index=True)

            except Exception:
                st.error("Prediction failed.")
                st.code(traceback.format_exc())


# ============================================================
# TAB 3 — MONITORING
# ============================================================
def tab_monitoring():
    st.markdown('<div class="section-title">Model Monitoring</div>', unsafe_allow_html=True)
    st.caption("Production monitoring is designed around Gini, KS and PSI thresholds.")

    metrics = load_metrics().copy()

    if "roc_auc" in metrics.columns:
        metrics["gini"] = 2 * metrics["roc_auc"] - 1
    else:
        metrics["gini"] = np.nan

    if "ks" not in metrics.columns:
        metrics["ks"] = np.nan
    if "psi" not in metrics.columns:
        metrics["psi"] = np.nan

    thresholds = {"gini": 0.30, "ks": 0.20, "psi": 0.25}

    c1, c2, c3 = st.columns(3)
    with c1:
        max_gini = metrics["gini"].max() if metrics["gini"].notna().any() else np.nan
        metric_card("Gini", f"{max_gini:.3f}" if pd.notna(max_gini) else "—", f"Floor {thresholds['gini']:.2f}")
    with c2:
        max_ks = metrics["ks"].max() if metrics["ks"].notna().any() else np.nan
        metric_card("KS", f"{max_ks:.3f}" if pd.notna(max_ks) else "—", f"Floor {thresholds['ks']:.2f}")
    with c3:
        max_psi = metrics["psi"].max() if metrics["psi"].notna().any() else np.nan
        metric_card("PSI", f"{max_psi:.3f}" if pd.notna(max_psi) else "—", f"Ceiling {thresholds['psi']:.2f}")

    st.markdown('<div class="section-title">Gini by Model</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_bar(x=metrics["model"], y=metrics["gini"], name="Gini")
    fig.add_hline(y=thresholds["gini"], line_dash="dash", annotation_text="Gini floor")
    plot_layout(fig, height=400)
    st.plotly_chart(fig, width='stretch', theme=None)

    alerts = metrics[
        (metrics["gini"] < thresholds["gini"])
        | (metrics["ks"].fillna(1) < thresholds["ks"])
        | (metrics["psi"].fillna(0) > thresholds["psi"])
    ]

    st.markdown('<div class="section-title">Monitoring Alerts</div>', unsafe_allow_html=True)

    if alerts.empty:
        st.markdown(
            '<div class="success-card">✓ All available model metrics are within configured thresholds.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="alert-card">⚠ {len(alerts)} model(s) breached at least one monitoring threshold.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(alerts, width='stretch', hide_index=True)

    st.markdown('<div class="section-title">Monitoring Table</div>', unsafe_allow_html=True)
    st.dataframe(metrics, width='stretch', hide_index=True)


# ============================================================
# TAB 4 — FRAUD
# ============================================================
def tab_fraud():
    st.markdown('<div class="section-title">Fraud Risk</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ui-card">
            <div class="card-title">🛡️ Fraud Risk Preview</div>
            <div class="card-subtitle">
                Fraud model integration is currently under development.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Fraud Probability", "XGBoost / LightGBM", "Applicant-level fraud probability.")
    with c2:
        card("Fraud Typology", "Policy-grounded", "Account takeover, synthetic ID, first-party fraud, etc.")
    with c3:
        card("Red Flags", "Rule IDs", "Policy citations and operational signals.")

    st.markdown('<div class="section-title">Planned Fraud Pipeline</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-title">LLM Explanation</div>', unsafe_allow_html=True)
    st.caption("The LLM explains the ML result; it does not replace the underlying credit-risk model.")

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        model_name = st.selectbox(
            "Model",
            MODEL_CHOICES,
            index=MODEL_CHOICES.index(DEFAULT_MODEL),
            key="explain_model",
        )
        raw = applicant_input("explain")
        run_llm = st.button(
            "Generate Explanation",
            type="primary",
            width='stretch',
            key="explain_run",
        )

    with right:
        st.markdown(
            """
            <div class="ui-card">
                <div class="card-title">LLM Architecture</div>
                <div class="card-subtitle">Grounded business explanation</div>
                <div style="margin-top:1rem;line-height:2;font-size:0.88rem;">
                    <div>① ML prediction</div><div>↓</div>
                    <div>② Probability</div><div>↓</div>
                    <div>③ Applicant features</div><div>↓</div>
                    <div>④ Hugging Face Router</div><div>↓</div>
                    <div>⑤ Business explanation</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_llm:
        with st.spinner("Scoring and generating explanation..."):
            try:
                result = score_one(model_name, raw)

                st.markdown(
                    f"""
                    <div class="ui-card">
                        <div style="display:flex;justify-content:space-between;">
                            <div>
                                <div class="card-title">Prediction</div>
                                <div style="font-size:1.5rem;font-weight:750;margin-top:0.2rem;">
                                    Class {result['predicted_class']}
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div class="card-subtitle">Confidence</div>
                                <div style="font-size:1.4rem;font-weight:750;">
                                    {result['confidence']:.2%}
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                proba_df = pd.DataFrame({
                    "class": list(result["class_probabilities"].keys()),
                    "probability": list(result["class_probabilities"].values()),
                })
                fig = px.bar(proba_df, x="class", y="probability", title="")
                fig.update_layout(yaxis_tickformat=".0%")
                plot_layout(fig, height=350)
                st.plotly_chart(fig, width='stretch', theme=None)

                explanation = explain_prediction(result, raw)

                st.markdown('<div class="section-title">Business Explanation</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ui-card">{explanation}</div>',
                    unsafe_allow_html=True,
                )

            except Exception:
                st.error("LLM pipeline failed.")
                st.code(traceback.format_exc())
    else:
        st.info("Enter applicant features and click **Generate Explanation**.")


# ============================================================
# TAB 6 — CHATBOT
# ============================================================
def tab_chatbot():
    st.header("🤖 Credit Risk Chatbot")

    st.caption(
        "Ask questions about policies, credit risk, or how the model works. "
        "Conversation history is kept for this session only."
    )

    if "chatbot_messages" not in st.session_state:
        st.session_state.chatbot_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your Credit Risk Assistant. "
                           "Ask me about policies, credit scoring, "
                           "or the ML models behind this dashboard.",
            }
        ]

    for msg in st.session_state.chatbot_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask about credit risk...", key="chatbot_input"):
        st.session_state.chatbot_messages.append(
            {"role": "user", "content": user_prompt}
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            try:
                if not HF_TOKEN:
                    raise RuntimeError("HUGGINGFACE_API_KEY is not set.")

                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chatbot_messages
                ]

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
                        "temperature": 0.3,
                    },
                    timeout=45,
                )
                response.raise_for_status()
                full_response = response.json()["choices"][0]["message"]["content"].strip()

            except Exception as e:
                full_response = f"❌ LLM call failed: {e}"

            placeholder.markdown(full_response)

        st.session_state.chatbot_messages.append(
            {"role": "assistant", "content": full_response}
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
elif page == "🤖 Chatbot":
    tab_chatbot()


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="
        margin-top:2rem;
        padding-top:1rem;
        border-top:1px solid rgba(148,163,184,0.3);
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