# Credit Risk Superset Dashboards

## What this produces

One dashboard — **Credit Risk — Model Monitoring** — with 7 charts:

| # | Chart | Type | Data source | Answers |
|---|---|---|---|---|
| 1 | Model Scoreboard (ROC AUC) | table | `v_model_scoreboard` | Which challenger wins? |
| 2 | Gini Over Time | line | `v_monitoring_trend` | Is the champion's Gini slipping? |
| 3 | KS Over Time | line | `v_monitoring_trend` | Is rank-ordering power stable? |
| 4 | PSI Heatmap | heatmap | `v_psi_heatmap` | Which models drifted, which month? |
| 5 | Active Alerts | table | `v_monitoring_alerts` | What needs a stakeholder ping? |
| 6 | Approval Distribution | pie | `v_approval_distribution` | Are approval volumes skewed? |
| 7 | Score Band by Approval | bar | `v_score_band` | Do decisions track credit score? |

## Run

```bash
# 1. Make sure Postgres + Superset are up (docker-compose / k8s)
# 2. Populate monitoring tables (once)
python -m data_science.src.credit_risk.etl_monitoring

# 3. Bootstrap the dashboard
python -m data_science.superset.create_dashboard
```

## Re-running

`create_dashboard.py` is idempotent — rerun any time you change `charts.yaml`, or after every ETL cycle. It updates existing charts/dashboard in place.

## Scheduling

- **Views + monitoring ETL**: nightly cron / Airflow, e.g. `0 3 * * *`.
- **Bot alerts**: on the `v_monitoring_alerts` view, checked every hour; posts to Slack via webhook.
- **Dashboard refresh**: Superset caches per-dataset; disable cache for `v_monitoring_latest` if you want live values.