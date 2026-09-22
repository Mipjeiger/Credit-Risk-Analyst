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
LLM_API_URL = ("https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct")

# Load ML model from Hugging Face Hub
all_files = list_repo_files(repo_id=REPO_ID)
model_files = [
    f for f in all_files if f.startswith("models/") and f.endswith(".joblib")
]
champion_model_path = hf_hub_download(repo_id=REPO_ID, filename=model_files[0])
model = joblib.load(champion_model_path)

# Load feature metadata
meta_path = hf_hub_download(
    repo_id=REPO_ID, filename="metadata/metadata.json"
)
with open(meta_path) as f:
    meta = json.load(f)

FEATURE_COLUMNS = meta["feature_columns"]

# ---------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------
def predict_credit_risk(*values):
    """Retrieves feature values and returns ML prediction results."""
    input_df = pd.DataFrame([dict(zip(FEATURE_COLUMNS, values))])
    input_df = input_df[FEATURE_COLUMNS]  # Ensure correct column order
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].max()
    return prediction, probability

def llm_explain(prediction, probability, feature_values):
    """Calls the LLM via Hugging Face Inference API for interpretation."""
    if not HF_API_TOKEN:
        return "⚠️ HF_TOKEN environment variable is not set in Space secrets."

    prompt = f"""
            You are a credit risk analyst. A model predicted class '{prediction}' with {probability:.2%} confidence based on the following features:
            {json.dumps(dict(zip(FEATURE_COLUMNS, feature_values)), indent=2)}
            Explain what this means for the applicant and why this prediction was made. 
            Keep it concise."""

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.3},
    }

    try:
        response = requests.post(
            LLM_API_URL, headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            generated = result[0].get("generated_text", "")
            # Clean up template prompt output if returned
            if "<|im_start|>assistant" in generated:
                generated = generated.split("<|im_start|>assistant")[-1]
            return generated.strip()
        return str(result)

    except Exception as e:
        return f"❌ Error calling LLM Inference API: {e}"

# ---------------------------------------------------------
# ZeroGPU Decorated Entrypoint
# ---------------------------------------------------------
@spaces.GPU
def run_prediction_pipeline(*values):
    """ZeroGPU allocates hardware dynamically when this function runs."""
    pred, prob = predict_credit_risk(*values)
    result_text = f"Prediction: {pred} (Confidence: {prob:.2%})"
    explanation = llm_explain(pred, prob, values)
    return result_text, explanation


# ---------------------------------------------------------
# Gradio Interface Build
# ---------------------------------------------------------
with gr.Blocks(title="Credit Risk Challenger Model") as demo:
    gr.Markdown("## Credit Risk Prediction & Interpretation")
    gr.Markdown(
        "Enter feature values to generate a credit risk assessment and an LLM summary."
    )

    inputs = []
    # Display input boxes for the primary features
    with gr.Row():
        for col in FEATURE_COLUMNS:
            with gr.Column():
                inputs.append(gr.Number(label=col, value=0.0))

    predict_btn = gr.Button("Evaluate Credit Risk", variant="primary")

    with gr.Row():
        output_pred = gr.Textbox(label="Prediction Result")
        output_explain = gr.Textbox(label="LLM Explanation", lines=6)

    # Bind execution directly to the @spaces.GPU function
    predict_btn.click(
        fn=run_prediction_pipeline,
        inputs=inputs,
        outputs=[output_pred, output_explain],
    )

if __name__ == "__main__":
    demo.launch()