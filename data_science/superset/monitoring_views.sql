-- ------------------------------------------------------------------
-- Base: current portfolio snapshot (one row per application)
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW credit_risk.v_portfolio AS
SELECT
    prospectid,
    approved_flag,
    credit_score,
    netmonthlyincome,
    age,
    gender,
    education,
    maritalstatus,
    first_prod_enq2,
    last_prod_enq2,
    cc_utilization,
    pl_utilization,
    tot_missed_pmnt,
    num_times_delinquent,
    max_delinquency_level,
    created_at
FROM credit_risk.data;

-- ------------------------------------------------------------------
-- Model scoreboard: parquet of model_metrics.csv loaded as a table
-- (see etl_metrics.py). One row per model.
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW credit_risk.v_model_scoreboard AS
SELECT
    model_name,
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    rank() OVER (ORDER BY roc_auc DESC) AS roc_auc_rank,
    rank() OVER (ORDER BY f1 DESC) AS f1_rank,
FROM credit_risk.model_metrics;

-- ------------------------------------------------------------------
-- Monitoring metrics (Gini / KS / PSI) computed nightly by a job.
-- Table: credit_risk.model_monitoring
-- Columns: model_name, scored_at, gini, ks, psi, gini_min, ks_min, psi_max
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW credit_risk.v_monitoring_latest AS
SELECT DISTINCT ON (model_name)
    model_name,
    scored_at,
    gini,
    ks,
    psi,
    gini_min,
    ks_min,
    psi_max,
    CASE
        WHEN gini < gini_min THEN 'GINI_LOW'
        WHEN ks < ks_min THEN 'KS_LOW'
        WHEN psi > psi_max THEN 'PSI_HIGH'
        ELSE 'OK'
    END AS status,
    (gini >= gini_min AND ks >= ks_min AND psi <= psi_max) AS is_healthy
FROM credit_risk.model_monitoring
ORDER BY model_name, scored_at DESC;

CREATE OR REPLACE VIEW credit_risk.v_monitoring_trend AS
SELECT
    model_name,
    scored_at,
    gini,
    ks,
    psi,
    gini_min,
    ks_min,
    psi_max
FROM credit_risk.model_monitoring
ORDER BY scored_at;

-- ------------------------------------------------------------------
-- PSI drift heatmap: month × model
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW credit_risk.v_psi_heatmap AS
SELECT
    date_trunc('month', scored_at)::date AS month,
    model_name,
    psi
FROM credit_risk.model_monitoring;

-- ------------------------------------------------------------------
-- Approval distribution: expected vs actual approval rates by model
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW credit_risk.v_approval_distribution AS
SELECT
    approved_flag,
    COUNT(*) AS applications,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct,
    AVG(credit_score) AS avg_credit_score,
    AVG(netmonthlyincome) AS avg_income
FROM credit_risk.data
GROUP BY approved_flag;

-- ------------------------------------------------------------------
-- Credit score bands × approval (for the "band × outcome" chart)
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW creidt_risk.v_score_band AS
SELECT
    approved_flag,
    CASE
        WHEN credit_score < 600 THEN '0-599'
        WHEN credit_score < 650 THEN '600-649'
        WHEN credit_score < 700 THEN '650-699'
        WHEN credit_score < 750 THEN '700-749'
        WHEN credit_score < 800 THEN '750-799'
        ELSE '800+'
    END AS score_band,
    COUNT(*) AS applications
FROM credit_risk.data
GROUP BY approved_flag, score_band;

-- ------------------------------------------------------------------
-- Alerts: rows the bot / dashboard should highlight
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW credit_risk.v_monitoring_alerts AS
SELECT
    model_name,
    scored_at,
    status,
    gini, 
    gini_min,
    ks, 
    ks_min,
    psi, 
    psi_max
FROM credit_risk.v_monitoring_latest
WHERE status <> 'OK';