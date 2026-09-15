#!/usr/bin/env bash
# =============================================================================
# ml_production.sh
# -----------------------------------------------------------------------------
# End-to-end ML pipeline for the Credit Risk Analyst project.
#
# Usage:
#   ./ml_production.sh [command]
#
# Commands:
#   train      Train all models + bundle, log to MLflow
#   evaluate   Evaluate the saved bundle, log metrics to MLflow
#   register   Promote the best run to MLflow Model Registry (Production)
#   all        train -> evaluate -> register (default)
#   clean      Remove local model artifacts
#   help       Show this help

# --- Step by Step to run the ML pipeline ---

# make it executable (once)
# chmod +x ml_production.sh

# full pipeline: train → evaluate → register
# ./ml_production.sh all

# individual steps
# ./ml_production.sh train
# ./ml_production.sh evaluate
# ./ml_production.sh register

# housekeeping
# ./ml_production.sh clean
# ./ml_production.sh help
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Python interpreter (override with PYTHON=... if needed)
PYTHON="${PYTHON:-python3}"

# MLflow
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-credit_risk_ml}"

# Paths
DATA_FILE="${PROJECT_ROOT}/credit_risk_production/database/data/merged_credit_risk_data.parquet"
ARTIFACT_DIR="${PROJECT_ROOT}/credit_risk_production/database/LLM/outputs_llm/model_artifacts_mlflow"
LOG_DIR="${PROJECT_ROOT}/credit_risk_production/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/ml_production_${TIMESTAMP}.log"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }
die()  { echo "❌ $*" >&2; exit 1; }

banner() {
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "  $*"                                                          | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
}

check_env() {
    mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"

    command -v "${PYTHON}" >/dev/null 2>&1 \
        || die "Python interpreter '${PYTHON}' not found."

    "${PYTHON}" - <<'PY' || die "Required Python packages missing."
import importlib, sys
mods = ["pandas", "numpy", "sklearn", "joblib", "mlflow",
        "xgboost", "pyarrow"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("Missing packages:", ", ".join(missing))
    sys.exit(1)
PY

    [[ -f "${DATA_FILE}" ]] \
        || die "Training data not found: ${DATA_FILE}"

    log "Python      : $(${PYTHON} --version 2>&1)"
    log "Project root: ${PROJECT_ROOT}"
    log "Data file   : ${DATA_FILE}"
    log "Artifacts   : ${ARTIFACT_DIR}"
    log "MLflow URI  : ${MLFLOW_TRACKING_URI}"
    log "Experiment  : ${MLFLOW_EXPERIMENT_NAME}"
}

# -----------------------------------------------------------------------------
# Pipeline steps
# -----------------------------------------------------------------------------
step_train() {
    banner "STEP 1/3 — TRAIN"
    cd "${PROJECT_ROOT}/credit_risk_production"
    "${PYTHON}" -m ml.train 2>&1 | tee -a "${LOG_FILE}"
    log "✅ Training complete."
}

step_evaluate() {
    banner "STEP 2/3 — EVALUATE"
    [[ -f "${ARTIFACT_DIR}/model_bundle.joblib" ]] \
        || die "No model bundle found. Run 'train' first."
    cd "${PROJECT_ROOT}/credit_risk_production"
    "${PYTHON}" -m ml.evaluate 2>&1 | tee -a "${LOG_FILE}"
    log "✅ Evaluation complete."
}

step_register() {
    banner "STEP 3/3 — REGISTER"
    cd "${PROJECT_ROOT}/credit_risk_production"
    "${PYTHON}" -m ml.register 2>&1 | tee -a "${LOG_FILE}"
    log "✅ Registration complete."
}

step_clean() {
    banner "CLEAN"
    if [[ -d "${ARTIFACT_DIR}" ]]; then
        rm -f "${ARTIFACT_DIR}/model_bundle.joblib" \
              "${ARTIFACT_DIR}/model_metrics.csv" \
              "${ARTIFACT_DIR}"/cm_*.npz
        log "🧹 Removed local model artifacts from ${ARTIFACT_DIR}"
    else
        log "Nothing to clean."
    fi
}

usage() {
    grep -E '^#( |$)' "$0" | sed 's/^# \?//'
}

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
main() {
    local cmd="${1:-all}"

    case "${cmd}" in
        train)    check_env; step_train ;;
        evaluate) check_env; step_evaluate ;;
        register) check_env; step_register ;;
        all)      check_env; step_train; step_evaluate; step_register ;;
        clean)    step_clean ;;
        help|-h|--help) usage ;;
        *)        echo "Unknown command: ${cmd}"; usage; exit 1 ;;
    esac

    log "🎉 Done: ${cmd}"
}

main "$@"