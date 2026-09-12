import os
import json
import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import joblib
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from huggingface_hub import InferenceClient
from groq import Groq

# Configuration
BASE_PATH = Path(__file__).resolve().parents[1]
LLM_PATH = BASE_PATH / "database" / "LLM"
ENV_PATH = BASE_PATH / ".env"
load_dotenv(ENV_PATH)

class CreditRiskRAG:
    def __init__(self, base_dir: Path = LLM_PATH):
        self.base = Path(base_dir)
        self.cfg = json.loads((self.base / "outputs_llm" / "rag_config.json").read_text())
        self.manifest = json.loads((self.base / "outputs_llm" / "pipeline_manifest.json").read_text())
        self.model_bundle = joblib.load(self.base / "outputs_llm" / "model_artifacts" / "model_bundle.joblib")
        self.model_features = json.loads(self.base / "outputs_llm" / "model_artifacts" / "model_features.json").read_text()
        self.models = self.model_bundle['models']
        self.scaler = self.model_bundle['scaler']
        self.label_encoders = self.model_bundle.get('label_encoders', {})

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.cfg['embed_model'],
            encoder_kwargs={"normalize_embeddings": True}
        )

        chroma_dir = self.base / "chroma_store"
        self.customer_store = Chroma(
            collection_name=self.cfg['collections']['customers'],
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir)
        )
        self.policy_store = Chroma(
            collection_name=self.cfg['collections']['policies'],
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir)
        )

        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        self.hf_client = InferenceClient(token=hf_key) if hf_key else None
        self.groq_client = Groq(api_key=groq_key) if groq_key else None