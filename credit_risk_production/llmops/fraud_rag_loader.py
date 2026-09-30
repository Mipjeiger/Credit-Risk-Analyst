from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from groq import Groq
from openai import OpenAI

# -------------------------------------
# Configuration
# -------------------------------------
BASE_PATH = Path(__file__).resolve().parents[1]
LLM_PATH = BASE_PATH / "database" / "LLM"
ENV_PATH = BASE_PATH / ".env"
NINEROUTER_BASE_URL = os.getenv("NINEROUTER_BASE_URL")
NINEROUTER_API_KEY = os.getenv("NINEROUTER_API_KEY")
NINEROUTER_MODEL = "sragent"

load_dotenv(ENV_PATH)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FraudRiskRAG:
    def __init__(self, base_dir: Path = LLM_PATH, bundle_path: Optional[Path] = None):
        """Initialize the FraudRiskRAG with the base directory and optional bundle path."""
        self.base = Path(base_dir)

        # --- config and manifest files ---
        cfg_path = self.base / "outputs_llm" / "rag_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"RAG config file not found at {cfg_path}")
        self.cfg = json.loads((self.base / "outputs_llm" / "rag_config.json").read_text())

        manfiest_path = self.base / "outputs_llm" / "pipeline_manifest.json"
        if not manfiest_path.exists():
            raise FileNotFoundError(f"Pipeline manifest file not found at {manfiest_path}")
        self.manifest = json.loads((self.base / "outputs_llm" / "pipeline_manifest.json").read_text())

        # --- model bundle ---
        if bundle_path is None:
            bundle_path = self.base / "outputs_llm" / "model_artifacts" / "model_bundle.joblib"
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Model bundle file not found at {self.bundle_path}")

        bundle = joblib.load(self.bundle_path)
        self.models = bundle["models"]
        self.scaler         = bundle["scaler"]
        self.label_encoders = bundle.get("label_encoders", {})
        self.model_features = bundle["feature_columns"]
        self.class_labels   = bundle.get("class_labels", [0, 1])

        # ---- embeddings + vector store ----
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": "cpu"},
        )

        # --- Chroma vector store ---
        chroma_dir = self.base / "chroma_store"
        collections = self.cfg["collections"]

        self.customer_store = Chroma(
            collection_name=collections["fraud_cases"],
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir)
        )
        self.legit_store = Chroma(
            collection_name=collections["legit_cases"],
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir)
        )
        self.policy_store = Chroma(
            collection_name=collections["policies"],
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir)
        )

        # --- LLM clients ---
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        ninerouter_key = os.getenv("NINEROUTER_API_KEY")
        self.hf_client = InferenceClient(token=hf_key) if hf_key else None
        self.groq_client = InferenceClient(token=groq_key) if groq_key else None
        self.ninerouter_client = OpenAI(base_url=NINEROUTER_BASE_URL, api_key=NINEROUTER_API_KEY) if ninerouter_key else None

        logger.info(
            "FraudRiskRAG ready | models=%d | features=%d | bundle=%s",
            len(self.models), len(self.model_features), self.bundle_path,
        )

    # -------------------- Scoring and Prediction Methods --------------------
    def score(self, row: pd.Series | dict, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Single-row fraud prediction"""
        if isinstance(row, dict):
            row = pd.Series(row)

        if model_name is None:
            model_name = next(iter(self.models))
        if model_name not in self.models:
            raise KeyError(f"Unknown fraud model '{model_name}', available models: {list(self.models)}")

        model = self.models[model_name] # get the model object

        # 1. Align features with the model's expected features
        X = pd.DataFrame([row.reindex(self.model_features).values], columns=self.model_features)

        # 2. Encode categorical features using existing label encoders
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        # 3. Corce + Scale features
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

        # 4. Predict probabilities
        pred = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0]
        classes = [int(c) for c in model.classes_]
        pos_idx = classes.index(1) if 1 in classes else 0

        return {
            "model_used": model_name,
            "predicted_flag": pred,
            "class_probabilities": {f"class_{c}": round(float(p), 4) for c, p in zip(classes, proba)},
            "primary_fraud_probability": float(proba[pos_idx])
        }

    # -------------------- LLM  --------------------
    def llm(self, messages, max_tokens: int = 512, temperature: float = 0.3) -> Dict[str, str]:
        """HF Inference -> Groq -> Ninerouter Fallback"""
        # Huggingface Inference API
        if self.hf_client:
            try:
                r = self.hf_client.chat.completions.create(
                    model=self.cfg["llm"]["hf_model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return {"text": r.choices[0].message.content, "provider": "huggingface"}
            except Exception as e:
                logger.error(f"Huggingface Inference failed: {e}")

        # Groq Inference API
        if self.groq_client:
            try:
                r = self.groq_client.chat.completions.create(
                    model=self.cfg["llm"]["groq_model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return {"text": r.choices[0].message.content, "provider": "groq"}
            except Exception as e:
                logger.error(f"Groq Inference failed: {e}")

        # Ninerouter Fallback
        if self.ninerouter_client:
            try:
                r = self.ninerouter_client.chat.completions.create(
                    model=NINEROUTER_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"num_ctx": 16384},
                    stream=True
                )
                return {"text": r.choices[0].message.content, "provider": "ninerouter"}
            except Exception as e:
                logger.error(f"Ninerouter Inference failed: {e}")

        raise RuntimeError("No LLM provider available. Please check your API keys and configuration.")

    #  -------------------- Decide --------------------
    def decide(self, row: pd.Series | dict, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Full fraud decision pipeline: ML Score -> AUTO_ALLOW/AUTO_BLOCK/RAG_LLM"""
        if isinstance(row, dict):
            row = pd.Series(row)

        # Get the ML score
        ml = self.score(row, model_name=model_name)
        fp = ml["primary_fraud_probability"]
        th = self.cfg["thresholds"]

        # AUTO ALLOW
        if fp < th["auto_allow_below"]:
            return {
                "decision_route": "AUTO_ALLOW",
                "provider": "ML_Policy_Engine",
                **ml,
                "fraud_probability": fp,
                "fraud_probability_100": round(fp * 100, 2),
                "fraud_band": "Low",
                "typology": "unknown",
                "recommended_action": "allow",
                "confidence": fp,
                "needs_human_review": True
            }

        # AUTO BLOCK
        if fp > th["auto_block_above"]:
            return {
                "decision_route": "AUTO_BLOCK",
                "provider": "ML_Policy_Engine",
                **ml,
                "fraud_probability": fp,
                "fraud_probability_100": round(fp * 100, 2),
                "fraud_band": "Critical",
                "typology": "unknown",
                "recommended_action": "block",
                "confidence": fp,
                "needs_human_review": False,
            }

        # RAG + LLM Decision
        q = row.to_json()
        fraud_cases = self.customer_store.similarity_search(q, k=th["k_customers"])
        legit_cases = self.legit_store.similarity_search(q, k=th["k_customers"])
        policies = self.policy_store.similarity_search(q, k=th["k_policies"])

        ctx_fraud = "\n\n".join(d.page_content[:600] for d in fraud_cases)
        ctx_legit = "\n\n".join(d.page_content[:600] for d in legit_cases)
        ctx_pol = "\n\n".join(f"[{d.metadata.get('doc_type', 'policy')}] {d.page_content}" for d in policies)

        user_msg = (
            "ML FRAUD OUTPUT:\n"
            f" - model_used: {ml['model_used']}\n"
            f" - fraud_probability: {fp:.4f}\n"
            f" - predicted_flag: {ml['predicted_flag']}\n"
            f" - class_probabilities: {ml['class_probabilities']}\n\n"
            f"CUSTOMER FEATURES:\n{row.to_dict()}\n\n"
            f"SIMILAR FRAUD CASES:\n{ctx_fraud}\n\n"
            f"SIMILAR LEGITIMATE CASES:\n{ctx_legit}\n\n"
            f"RETRIEVED POLICIES:\n{ctx_pol}\n\n"
            f"ALLOWED ACTIONS: {self.cfg['enums']['actions']}\n"
            f"ALLOWED TYPOLOGIES: {self.cfg['enums']['typologies']}\n\n"
            "Return STRICT JSON only."
        )

        out = self.llm(
            messages=[
                {"role": "system", "content": self.cfg["system_prompt"]},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=self.cfg["llm"]["max_tokens"],
            temperature=self.cfg["llm"]["temperature"]
        )

        # Strip possible json fences and parse
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
            "fraud_probability": fp,
            "fraud_probability_100": round(fp * 100, 2),
            **parsed
        }

    # -------------------- Convenience --------------------
    def model_names(self) -> list[str]:
        return sorted(self.models.keys())