import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env next to this file, regardless of CWD.
ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

_llm: Optional[ChatOpenAI] = None


def get_llm() -> ChatOpenAI:
    """Return the shared LLM instance, creating it only when needed."""
    global _llm

    if _llm is not None:
        return _llm

    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Create a .env file next to config.py "
            "and add DEEPSEEK_API_KEY=your_actual_deepseek_api_key."
        )

    # Low temperature keeps intent routing, SQL, and chart config generation
    # deterministic and reproducible instead of creatively varied.
    _llm = ChatOpenAI(
        api_key=api_key,
        base_url=api_base,
        model=model_name,
        temperature=0.1,
    )
    return _llm
