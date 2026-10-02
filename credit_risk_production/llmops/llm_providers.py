import os
import logging
import requests

logger = logging.getLogger(__name__)

# Define url of the LLM providers
HF_CHAT_URL   = "https://router.huggingface.co/v1/chat/completions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

def _post(url, token, model, messages, max_tokens, temperature):
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        timeout=45
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def call_llm(messages, *, hf_model, groq_model, niner_model, max_tokens=512, temperature=0.3):
    """HF → Groq → 9router. Returns {'text':..., 'provider':...}."""
    # Read key from environment variables
    hf_key     = os.getenv("HUGGINGFACE_API_KEY")
    groq_key   = os.getenv("GROQ_API_KEY")
    niner_url  = os.getenv("NINEROUTER_BASE_URL")
    niner_key  = os.getenv("NINEROUTER_API_KEY")

    if hf_key:
        try:
            return {"text": _post(
                HF_CHAT_URL,
                hf_key,
                hf_model,
                messages,
                max_tokens,
                temperature
            ), 
            "provider": "Hugging Face"}
        except Exception as e:
            logger.error("HuggingFace router failed: %s — falling back to Groq", e)

    if groq_key:
        try:
            return {"text": _post(
                GROQ_CHAT_URL,
                groq_key,
                groq_model,
                messages,
                max_tokens,
                temperature
            ), 
            "provider": "Groq"}
        except Exception as e:
            logger.error("Groq failed: %s — falling back to 9router", e)

    if niner_url and niner_key:
        try:
            url = niner_url.rstrip("/")
            if not url.endswith("/chat/completions"):
                url += "/chat/completions"
            return {"text": _post(
                url,
                niner_key,
                niner_model,
                messages,
                max_tokens,
                temperature
            ), 
            "provider": "9router"}
        except Exception as e:
            logger.error("Ninerouter Inference failed: %s", e)

    raise RuntimeError(
        "No LLM provider available. Checked HF, Groq, 9router — "
        "all missing keys or all requests failed."
    )