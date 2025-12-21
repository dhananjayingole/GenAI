from dotenv import load_dotenv
import os

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env")

# ✅ CORRECT parameter name
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=128,
    temperature=0.5,
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke(
    [HumanMessage(content="What is the capital of India?")]
)

print("\n--- Hugging Face ChatModel Response ---\n")
print(response.content)
