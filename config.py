import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Verify API key
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not api_key:
    print("\n[ERROR] DEEPSEEK_API_KEY is not set in the .env file.")
    print("Please open the .env file in the workspace and set your DeepSeek API key.")
    sys.exit(1)

# Instantiate the shared LLM instance
# We configure a temperature of 0.75 for balanced creativity and consistency.
llm = ChatOpenAI(
    api_key=api_key,
    base_url=api_base,
    model=model_name,
    temperature=0.75,
)
