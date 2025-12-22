from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# HumanMessage(content="I want to request a refund for my order #12345.")
# AIMessage(content="Your refund request for order #12345 has been initiated. It will be processed in 3-5 business days.")

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HF_TOKEN,
    task="conversational",     # ✅ FIXED
    max_new_tokens=150,
    temperature=0.5,
)

chat_model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))   # ✅ FIXED

    result = chat_model.invoke(chat_history)

    chat_history.append(AIMessage(content=result.content))  # ✅ FIXED

    print("AI:", result.content)
