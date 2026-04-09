import os
import openai
import httpx
import google.generativeai as genai


def load_model(
    name,
    API_BASE="Default",
    API_KEY="Default",
    organization=False,
    provider="auto",
):
    client = None

    name_l = (name or "").lower()
    provider = (provider or "auto").lower()

    def _openai_client(base_url, api_key, timeout_s=3000.0, headers=None, org=None):
        http_client = httpx.Client(
            headers=headers or {},
            timeout=httpx.Timeout(timeout_s, read=60.0, write=60.0, connect=10.0),
        )
        return openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            organization=org,
            http_client=http_client,
        )

    if provider == "openrouter":
        API_BASE = (
            "https://openrouter.ai/api/v1"
            if API_BASE == "Default"
            else API_BASE
        )
        API_KEY = os.getenv("OPENROUTER_API_KEY") if API_KEY == "Default" else API_KEY
        headers = {}
        if os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER")
        if os.getenv("OPENROUTER_APP_TITLE"):
            headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE")
        client = _openai_client(API_BASE, API_KEY, headers=headers)
        return client

    if provider == "openai" or (provider == "auto" and "gpt" in name_l):
        API_BASE = "https://api.openai.com/v1" if API_BASE == "Default" else API_BASE
        API_KEY = os.getenv("OPENAI_API_KEY") if API_KEY == "Default" else API_KEY
        if organization:
            client = _openai_client(
                API_BASE,
                API_KEY,
                timeout_s=300.0,
                org=os.getenv("OPENAI_ORG_KEY"),
            )
        else:
            client = _openai_client(API_BASE, API_KEY, timeout_s=3000.0)
    elif provider == "gemini" or (provider == "auto" and "gemini" in name_l):
        API_KEY = os.getenv("GOOGLE_API") if API_KEY == "Default" else API_KEY
        genai.configure(api_key=API_KEY)
        generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
            }
        thresh = "BLOCK_MEDIUM_AND_ABOVE"
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": thresh
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": thresh
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": thresh
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": thresh
            },
        ]
        client = genai.GenerativeModel(
            model_name=name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
    elif provider == "auto" and ("llama" in name_l or "mixtral" in name_l or "mistral" in name_l):
        # Default hosted open-model path: OpenRouter (OpenAI-compatible)
        API_BASE = (
            "https://openrouter.ai/api/v1"
            if API_BASE == "Default"
            else API_BASE
        )
        API_KEY = os.getenv("OPENROUTER_API_KEY") if API_KEY == "Default" else API_KEY
        headers = {}
        if os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER")
        if os.getenv("OPENROUTER_APP_TITLE"):
            headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE")
        client = _openai_client(API_BASE, API_KEY, headers=headers)
    return client

