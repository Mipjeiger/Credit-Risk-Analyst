import json
import os
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import requests
import spaces
from huggingface_hub import hf_hub_download, list_repo_files

# ---------------------------------------------------------
# Configuration & Asset Loading
# ---------------------------------------------------------
REPO_ID = "Mipjeiger/credit-risk-challengers"
HF_API_TOKEN = os.getenv("HF_TOKEN")
LLM_API_URL = "https://router.huggingface.co/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# ---------------------------------------------------------
# Load all six models from models/*.joblib
# ---------------------------------------------------------
model_files = sorted(
    f for f in list_repo_files(repo_id=REPO_ID)
    if f.startswith("models/") and f.endswith(".joblib")
)

MODELS = {}
for f in model_files:
    stem = f.split("/")[-1].removesuffix(".joblib")
    name = stem.replace("_", " ").title()
    MODELS[name] = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=f))

MODEL_CHOICES = list(MODELS.keys())
DEFAULT_MODEL = "Gradient Boosting" if "Gradient Boosting" in MODELS else MODEL_CHOICES[0]

# ---------------------------------------------------------
# Load metadata (feature order)
# ---------------------------------------------------------
meta_path = hf_hub_download(repo_id=REPO_ID, filename="metadata/metadata.json")
with open(meta_path) as f:
    META = json.load(f)
FEATURE_COLUMNS = META["feature_columns"]

# ---------------------------------------------------------
# Load preprocessors (scaler + encoders)
# ---------------------------------------------------------
SCALER = joblib.load(hf_hub_download(repo_id=REPO_ID, filename="metadata/scaler.joblib"))
LABEL_ENCODERS = joblib.load(hf_hub_download(repo_id=REPO_ID, filename="metadata/label_encoders.joblib"))

print("=" * 60)
for name, m in MODELS.items():
    print(f"{name:25s} classes={getattr(m, 'classes_', None)} "
          f"n_features={getattr(m, 'n_features_in_', None)}")
print(f"Features: {len(FEATURE_COLUMNS)}")
print("=" * 60)

# ---------------------------------------------------------
# Prediction & LLM Explanation Functions
# ---------------------------------------------------------
def predict_credit_risk(model_name, *values):
    """Retrieves feature values and returns ML prediction results."""
    model = MODELS[model_name]

    input_df = pd.DataFrame([dict(zip(FEATURE_COLUMNS, values))])
    input_df = input_df[FEATURE_COLUMNS]  # Ensure correct order

    # Encode categorical features
    for col, le in LABEL_ENCODERS.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str).map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )

    # Coerce to numeric and fill NaN
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale features
    input_scaled = pd.DataFrame(SCALER.transform(input_df), columns=input_df.columns)

    # Class prediction and probability
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    classes = float(proba.max())
    confidence = float(proba.max())
    class_probs = {f"class_{int(c)}": round(float(p), 4) for c, p in zip(classes, proba)}

    return int(prediction), confidence, class_probs

def llm_explain(prediction, probability, feature_values):
    """Calls the LLM via Hugging Face Inference API for interpretation."""
    if not HF_API_TOKEN:
        return "⚠️ HF_TOKEN environment variable is not set in Space secrets."

    prompt = (
        f"You are a credit risk analyst. A model predicted class '{prediction}' "
        f"with {probability:.2%} confidence based on these features:\n"
        f"{json.dumps(dict(zip(FEATURE_COLUMNS, feature_values)), indent=2)}\n\n"
        f"Explain what this means for the applicant and why this prediction was made. "
        f"Keep it concise."
    )

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 256,
        "temperature": 0.3
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"❌ Error calling LLM API: {str(e)}"

# ---------------------------------------------------------
# ZeroGPU Decorated Entrypoint - run inference pipeline
# ---------------------------------------------------------
@spaces.GPU
def run_prediction_pipeline(model_name, *values):
    """ZeroGPU allocates hardware dynamically when this function runs."""
    pred, conf, probs = predict_credit_risk(model_name, *values)
    pred_text = (
        f"Model: {model_name}\n"
        f"Prediction: class {pred}\n"
        f"Confidence: {conf:.2%}\n"
        f"Class probabilities: {json.dumps(probs, indent=2)}"
    )
    explanation = llm_explain(model_name, pred, conf, values)
    return pred_text, explanation

# ---------------------------------------------------------
# Gradio Interface Build
# ---------------------------------------------------------
with gr.Blocks(title="Credit Risk — Model Challenger") as demo:
    gr.Markdown("## Credit Risk Prediction & Interpretation")

    model_dropdown = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Model")

    inputs = []
    with gr.Row():
        for col in FEATURE_COLUMNS:
            with gr.Column():
                inputs.append(gr.Number(label=col, value=0.0))

    predict_btn = gr.Button("Evaluate Credit Risk", variant="primary")

    with gr.Row():
        output_pred = gr.Textbox(label="Prediction Result", lines=8)
        output_explain = gr.Textbox(label="LLM Explanation", lines=8)

    predict_btn.click(
        fn=run_prediction_pipeline,
        inputs=[model_dropdown, *inputs],
        outputs=[output_pred, output_explain],
    )

if __name__ == "__main__":
    demo.launch()