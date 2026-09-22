import json
from pathlib import Path
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, text
from .monitoring import gini, ks, psi, DEFAULT_THRESHOLDS
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

"""Load model_metrics.csv and computed Gini/KS/PSI into Postgres for Superset."""

# Configuration
PROJECT = Path(__file__).resolve().parents[3] / "credit_risk_production"
MODEL_DIR = PROJECT / "models" / "credit_risk"
META = json.loads((MODEL_DIR / "metadata_credit_risk" / "metadata.json").read_text())
PARQUET = PROJECT / "database" / "data" / "merged_credit_risk_data.parquet"
ENV_PATH = Path(__file__).resolve().parents[3] / "data_science" / "superset" / ".env"

# Load environment variables
load_dotenv(ENV_PATH)

# Postgresql connection
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = "localhost"  # Use localhost for local development; use "postgres" for Docker
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

POSTGRES_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

def main() -> None:
    engine = create_engine(POSTGRES_URL, pool_pre_ping=True)

    # --- model metrics ---
    metrics = pd.read_csv(MODEL_DIR / "metrics_credit_risk" / "model_metrics.csv")
    metrics.columns = [c.strip().lower().replace(" ", "_") for c in metrics.columns]
    metrics = metrics.rename(columns={"model": "model_name"})

    # Metrics to sql
    metrics.to_sql(
        "model_metrics", 
        engine, 
        schema="credit_risk",
        if_exists="replace",
        index=False
        )

    # --- model monitoring (Gini/KS/PSI per model)
    df = pd.read_parquet(PARQUET)
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if c == "Approved_Flag":
            continue
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    # Split into X and y -> train test split
    y = LabelEncoder().fit_transform(df["Approved_Flag"].astype(str))
    X = df[META["feature_columns"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cls = META["class_labels"][-1]
    y_bin = (y_test == cls).astype(int)

    rows = []
    for name, fname in META["model_files"].items():
        model = joblib.load(MODEL_DIR / "ml_credit_risk" / fname)
        p_train = model.predict_proba(X_train)[:, cls]
        p_test = model.predict_proba(X_test)[:, cls]

        # Append metrics
        rows.append({
            "model_name": name,
            "scored_at": pd.Timestamp.now(),
            "gini": gini(y_bin, p_test),
            "ks": ks(y_bin, p_test),
            "psi": psi(p_train, p_test),
            **DEFAULT_THRESHOLDS
        })

    mon = pd.DataFrame(rows)
    mon.to_sql(
        "model_monitoring",
        engine,
        schema="credit_risk",
        if_exists="append",
        index=False
    )
    print(f"✅ Loaded {len(metrics)} metric rows, {len(mon)} monitoring rows into Postgres.")

if __name__ == "__main__":
    main()