import pandas as pd

"""Shared feature engineering for ML training + serving"""

FEATURE_GROUPS = {
    "trade_lines": [
        "Total_TL", "Tot_Closed_TL", "Tot_Active_TL",
        "Total_TL_opened_L6M", "Tot_TL_closed_L6M",
        "pct_tl_open_L6M", "pct_tl_closed_L6M",
        "pct_active_tl", "pct_closed_tl",
        "Total_TL_opened_L12M", "Tot_TL_closed_L12M",
        "pct_tl_open_L12M", "pct_tl_closed_L12M",
    ],
    "product_mix": [
        "Auto_TL", "CC_TL", "Consumer_TL", "Gold_TL",
        "Home_TL", "PL_TL", "Secured_TL", "Unsecured_TL", "Other_TL",
    ],
    "payment_behavior": ["Tot_Missed_Pmnt", "Age_Oldest_TL", "Age_Newest_TL"],
    "delinquency": [
        "time_since_recent_payment", "time_since_first_deliquency",
        "time_since_recent_deliquency", "num_times_delinquent",
        "max_delinquency_level", "max_recent_level_of_deliq",
        "num_deliq_6mts", "num_deliq_12mts", "num_deliq_6_12mts",
        "max_deliq_6mts", "max_deliq_12mts",
        "num_times_30p_dpd", "num_times_60p_dpd",
        "num_std", "num_std_6mts", "num_std_12mts",
        "num_sub", "num_sub_6mts", "num_sub_12mts",
        "num_dbt", "num_dbt_6mts", "num_dbt_12mts",
        "num_lss", "num_lss_6mts", "num_lss_12mts",
        "recent_level_of_deliq",
    ],
    "enquiries": [
        "tot_enq", "CC_enq", "CC_enq_L6m", "CC_enq_L12m",
        "PL_enq", "PL_enq_L6m", "PL_enq_L12m",
        "time_since_recent_enq", "enq_L12m", "enq_L6m", "enq_L3m",
    ],
    "demographics": ["MARITALSTATUS"],
}

DERIVED_FEATURES = [
    "delinq_velocity", "enq_velocity", "unsecured_ratio",
    "active_ratio", "recent_open_ratio", "dpd_score", "asset_class_score",
]

def _sd(a, b):
    try:
        return float(a) / float(b) if pd.notna(a) and pd.notna(b) and b not in (0, None) else 0.0
    except Exception:
        return 0.0

def add_derived_features(row: pd.Series) -> pd.Series:
    r = row.copy()

    # Derived features
    r["delinq_velocity"]   = _sd(row.get("num_deliq_6mts", 0), (row.get("num_deliq_12mts", 0) or 0) + 1)
    r["enq_velocity"]      = _sd(row.get("enq_L3m", 0), (row.get("enq_L12m", 0) or 0) + 1)
    r["unsecured_ratio"]   = _sd(row.get("Unsecured_TL", 0), (row.get("Total_TL", 0) or 0) + 1)
    r["active_ratio"]      = _sd(row.get("Tot_Active_TL", 0), (row.get("Total_TL", 0) or 0) + 1)
    r["recent_open_ratio"] = _sd(row.get("Total_TL_opened_L6M", 0), (row.get("Total_TL", 0) or 0) + 1)
    r["dpd_score"]         = float(row.get("num_times_30p_dpd", 0) or 0) + 2 * float(row.get("num_times_60p_dpd", 0) or 0)
    r["asset_class_score"] = (float(row.get("num_sub", 0) or 0)
                              + 2 * float(row.get("num_dbt", 0) or 0)
                              + 3 * float(row.get("num_lss", 0) or 0))
    return r

def row_to_narrative(row: pd.Series, cid=None) -> str:
    """Convert a row of features to a narrative string"""
    parts = [f"CUSTOMER_PROFILE id={cid}"]

    for group, cols in FEATURE_GROUPS.items():
        lines = [f" {c}={row[c]}" for c in cols if c in row.index and pd.notna(row[c])]

        if lines:
            parts.append(f"[{group.upper()}]\n" + "\n".join(lines))
    df = [f" {c}={row[c]:.4f}" for c in DERIVED_FEATURES if c in row.index and pd.notna(row[c])]
    if df:
        parts.append(f"[DERIVED]\n" + "\n".join(df))

    return "\n".join(parts)