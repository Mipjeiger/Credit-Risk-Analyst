import os
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path
from dotenv import load_dotenv

"""Push the model "Qwen/Qwen2.5-Coder-32B-Instruct" to HuggingFace Hub."""

# Configuration
BASE_PATH = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_PATH / ".env"
LLM_OPS_PATH = BASE_PATH / "llmops" / "llm_artifacts"
load_dotenv(ENV_PATH)

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
REPO_ID = os.getenv("HF_HUB_REPO", "Mipjeiger/credit-risk-Qwen-2.5-32B-Instruct")
LOCAL_DIR = Path(os.getenv("LLM_PUSH_DIR", LLM_OPS_PATH))

def run():
    api = HfApi(token=HF_TOKEN)
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)

    # Create dir if it doesn't exist
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    (LOCAL_DIR / "README.md").write_text(
        "# Credit Risk LLM\nLoRA adapter / configs for meta-llama/Llama-3.2-3B-Instruct.\n"
    )
    (LOCAL_DIR / "adapter_config.json").write_text('{"base_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct"}')

    # Upload the folder to HuggingFace Hub
    upload_folder(
        repo_id=REPO_ID,
        folder_path=str(LOCAL_DIR),
        repo_type="model",
        token=HF_TOKEN,
        commit_message="Update credit-risk LLM artifacts"
    )
    print(f"✅ Success pushed to https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    run()