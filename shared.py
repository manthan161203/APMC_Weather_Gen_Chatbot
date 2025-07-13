# shared.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from configs import api_config
import os

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_config.openai_api_key)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
