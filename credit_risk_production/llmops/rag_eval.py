import os

import mlflow

from llmops.rag_loader import CreditRiskRAG

"""Evaluate RAG Retrieval quality + log to MLFlow"""

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_risk_llm")


def run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    rag = CreditRiskRAG()
    queries = [
        "What DPD threshold classifies an account as Sub-standard?",
        "Common red flags for synthetic identity fraud?",
        "Scorecard cutoff for manual review?",
    ]

    with mlflow.start_run(run_name="rag_eval"):
        for i, q in enumerate(queries):
            hits = rag.policy_store.similarity_search(q, k=5)
            mlflow.log_text(
                "\n\n".join(
                    f"[{h.metadata.get('doc_type')}] {h.page_content[:300]}"
                    for h in hits
                ),
                f"retrieval/q{i}.txt",
            )

        mlflow.log_param("embed_model", rag.cfg["embed_model"])
        mlflow.log_param("k_policies", rag.cfg["thresholds"]["k_policies"])
        print("✅ RAG evaluation logged")


if __name__ == "__main__":
    run()
