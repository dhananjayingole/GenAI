from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from langchain_google_genai import GoogleGenerativeAI

# Initialize Gemini LLM
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
)

# Invoke LLM (string → string)
response = llm.invoke(
    "Explain FastAPI in simple words for a beginner."
)

print("\n--- Gemini LLM Response ---\n")
print(response)
