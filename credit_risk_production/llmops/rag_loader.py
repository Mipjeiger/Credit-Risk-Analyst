import os
import json
import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import joblib
import logging
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from huggingface_hub import InferenceClient
from groq import Groq

# Configuration
BASE_PATH = Path(__file__).resolve().parents[1]
LLM_PATH = BASE_PATH / "database" / "LLM"
ENV_PATH = BASE_PATH / ".env"
load_dotenv(ENV_PATH)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class CreditRiskRAG:
    def __init__(self, base_dir: Path = LLM_PATH):
        self.base = Path(base_dir)
        self.cfg = json.loads((self.base / "outputs_llm" / "rag_config.json").read_text())
        self.manifest = json.loads((self.base / "outputs_llm" / "pipeline_manifest.json").read_text())
        self.model_bundle = joblib.load(self.base / "outputs_llm" / "model_artifacts" / "model_bundle.joblib")
        self.model_features = self.model_bundle['feature_columns']

        self.models = self.model_bundle['models']
        self.scaler = self.model_bundle['scaler']
        self.label_encoders = self.model_bundle.get('label_encoders', {})

        # Build a mapping dict for each encoder to handle unseen values safely
        self.label_mappings = {}
        for col, le in self.label_encoders.items():
            self.label_mappings[col] = {cls: idx for idx, cls in enumerate(le.classes_)}
            # Define a fallback index for unseen categories - using -1 oftern better than 0
            self.label_mappings[col]['<UNKNOWN>'] = -1

        self.X = self.model_bundle['feature_columns']
        self.y = "Approved_Flag"

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": "cpu"},
        )

        chroma_dir = self.base / "chroma_store"
        chroma_settings = Settings(anonymized_telemetry=False)
        self.customer_store = Chroma(
            collection_name=self.cfg['collections']['customers'],
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir),
            client_settings=chroma_settings
        )
        self.policy_store = Chroma(
            collection_name=self.cfg['collections']['policies'],
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir),
            client_settings=chroma_settings
        )

        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        self.hf_client = InferenceClient(token=hf_key) if hf_key else None
        self.groq_client = Groq(api_key=groq_key) if groq_key else None

    # ------ ML Scoring ------
    def score(self, row: pd.Series, model_name: str) -> Dict[str, Any]:
        """Score a single row using the specified string model chosen from the model bundle."""
        X = pd.DataFrame([row.reindex(self.model_features).values], columns=self.model_features)

        for col, le in self.label_encoders.items():
            if col in X.columns:
                try:
                    X[col] = le.transform(X[col].astype(str))
                except Exception:
                    logger.warning("LabelEncoder fallback for column %s", col)
                    X[col] = 0

        # Scale with the FITTED scaler
        X_scaled = pd.DataFrame(self.scaler.transform(X),columns=X.columns)

        # Predict 
        model = self.models[model_name]
        proba = model.predict_proba(X_scaled)[0]
        classes = list(model.classes_)
        pos_idx = list(model.classes_).index(1) if 1 in list(model.classes_) else 0

        class_probs = {
            f"class_{int(c)}_prob": round(float(p), 4)
            for c, p in zip(classes, proba)
        }

        return {
            "model_used": model_name,
            "predicted_flag": int(model.predict(X_scaled)[0]),
            "class_probabilities": class_probs,
            "primary_risk_probability": float(proba[pos_idx])
        }

    # ----- LLM Retrieval ------
    def _llm(self, messages, max_tokens=512, temperature=0.3):
        # Huggingface Inference API
        if self.hf_client:
            try:
                r = self.hf_client.chat.completions.create(
                    model=self.cfg["hf_model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return {
                    "text": r.choices[0].message.content,
                    "provider": "huggingface"
                }
            except Exception as e:
                logger.error(f"❌ Huggingface Failed: {e}")
                logger.info("Attempting fallback to Groq")

        # Fallback to Groq
        if self.groq_client:
            try:
                r = self.groq_client.chat.completions.create(
                    model=self.cfg["groq_model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return {
                    "text": r.choices[0].message.content,
                    "provider": "groq"
                }
            except Exception as e:
                logger.error(f"❌ Groq Failed: {e}")

        raise RuntimeError("❌ No LLM provider available. Please check your API keys and configuration.")

    # ----- Decide ------
    def decide(self, row: pd.Series) -> Dict[str, Any]:
        """
           Main function deciding on a credit risk scoring and LLM retrieval to decide
           Using LLM or action by ML models
           """
        ml = self.score(row)
        rp = ml["primary_risk_probability"]

        # Define logic auto reject
        if rp < self.cfg["thresholds"]["auto_reject_below"]:
            return {
                "decision_route": "AUTO_REJECT",
                "provider": "ML_Policy_Engine",
                **ml,
                "risk_score": rp,
                "risk_score_100": round(rp * 100, 2),
                "recommend_action": "Decline - risk score below cut-off"
            }

        # Define logic auto approve
        if rp > self.cfg["thresholds"]["auto_approve_above"]:
            return {
                "decision_route": "AUTO_APPROVE", 
                "provider": "ML_Policy_Engine",
                **ml, 
                "risk_score": rp, 
                "risk_score_100": round(rp * 100, 2),
                "recommended_action": "Approve - strong profile"}

        # RAG path
        q = row.to_json()
        cust = self.customer_store.similarity_search(q, k=self.cfg["thresholds"]["k_customers"])
        pol = self.policy_store.similarity_search(q, k=self.cfg["thresholds"]["k_policies"])
        ctx_c = "\n\n".join(d.page_content[:600] for d in cust)
        ctx_p = "\n\n".join(f"[{d.metadata.get('doc_type')}] {d.page_content}" for d in pol)

        user = f"""
        CUSTOMER:\n{row.to_dict()}\nML:\n{ml}\n

        SIMILAR:\n{ctx_c}\n\nPOLICIES:\n{ctx_p}\n
        Return ONLY JSON: risk_band, key_drivers, recommended_action, policy_refs, fraud_indicators, confidence."""

        out = self._llm([
            {"role": "system", "content": self.cfg["system_prompt"]},
            {"role": "user", "content": user}
        ])
        txt = re.sub(r"^```(json)?|```$", "", out["text"].strip(), flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {"raw_text": txt}

        return {
            "decision_route": "RAG_LLM",
            "provider": out["provider"],
            **ml,
            "risk_score": rp,
            "risk_score_100": round(rp * 100, 2),
            **parsed
        }