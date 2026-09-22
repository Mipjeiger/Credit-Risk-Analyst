import gradio as gr
import joblib
import os
import requests
import json
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

# Configuration
REPO_ID = "Mipjeiger/credit-risk-challengers"
HF_API_TOKEN = os.getenv("HF_TOKEN")
LLM_API_URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct"

# Load all ML models from huggingface hub repo
all_files = list_repo_files(repo_id=REPO_ID)
model_files = [f for f in all_files if f.startswith("models/") and f.endswith(".joblib")]
champion_model_path = hf_hub_download(repo_id=REPO_ID, filename=model_files[0])
model = joblib.load(champion_model_path)

# Download metadata to get feature columns
meta_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="metadata/metadata.json"
)
with open(meta_path) as f:
    meta = json.load(f)

# Feature columns
FEATURE_COLUMNS = meta["feature_columns"]

# Prediction on credit risk based on ML model
def predict_credit_risk(*values):
    """Retrieve feature values, returns prediction results"""
    input_df = pd.DataFrame([dict(zip(FEATURE_COLUMNS, values))])
    input_df = input_df[FEATURE_COLUMNS]  # Ensure correct order
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].max()
    return prediction, probability

# LLM Explanation (Using Inference API)
def llm_explain(prediction, probability, feature_values):
    """Calls the LLM via Inference API to explain the prediction"""
    if not HF_API_TOKEN:
        return "HF API token not set. Cannot call LLM for explanation."

    # Construct a prompt for the LLM
    prompt = f"""
           You are a credit risk analyst. 
           A model predicted class '{prediction}' with {probability:.2%} confidence based on the following features:
           {json.dumps(dict(zip(FEATURE_COLUMNS, feature_values)), indent=2)}
            Explain the reasoning what this means for the capplicant's credit risk and why this prediction was made.
            Keep it concise.
            """

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.3}}

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        # Handle the response if the format can be varied
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "Could not parse LLM response.").strip()
        return str(result)

    except Exception as e:
        return f"❌ Error calling LLM: {e}"

# Gradio Interface deployment
with gr.Blocks(title="Credit Risk Challenger Model") as demo:
    gr.Markdown("## Credit Risk Prediction & Interpretation")
    gr.Markdown("Enter feature values to see the model's prediction and an LLM-generated explanation.")

    # Create input components dynamically
    inputs = []
    with gr.Row():
        for i, col in enumerate(FEATURE_COLUMNS[:10]): # Display first 10 features for simplicity
            with gr.Column():
                inputs.append(gr.Number(label=col, value=0.0))

    with gr.Row():
        predict_btn = gr.Button("Predict")
        explain_btn = gr.Button("Explain Prediction", interactive=False) 

    output_pred = gr.Textbox(label="Prediction Result")
    output_explain = gr.Textbox(label="LLM Explanation", lines=5)

    # State to hold the last prediction details
    last_prediction = gr.State()

    def on_predict(*values):
        pred, prob = predict_credit_risk(*values)
        result_text = f"Prediction: {pred} (Confidence: {prob:.2%})"
        return result_text, (pred, prob, values), gr.update(interactive=True)

    def on_explain(state):
        if state is None:
            return "Please make a prediction first."
        pred, prob, values = state
        explanation = llm_explain(pred, prob, values)
        return explanation

    predict_btn.click(
        fn=on_predict,
        inputs=inputs,
        outputs=[output_pred, last_prediction, explain_btn]
    )

    explain_btn.click(
        fn=on_explain,
        inputs=[last_prediction],
        outputs=[output_explain]
    )

if __name__ == "__main__":
    demo.launch()