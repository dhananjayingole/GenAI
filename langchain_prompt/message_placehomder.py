from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --------------------------------------------------
# Initialize Hugging Face LLM
# --------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HF_TOKEN,
    task="conversational",
    max_new_tokens=200,
    temperature=0.4,
)

chat_model = ChatHuggingFace(llm=llm)

# --------------------------------------------------
# Prompt Template with Message Placeholder
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful and professional customer support assistant for a company. "
        "You help users with orders, refunds, delivery status, and complaints. "
        "Always remember previous conversation context and respond politely."
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{input}")
])

# --------------------------------------------------
# Create chain
# --------------------------------------------------
chain = prompt | chat_model

# --------------------------------------------------
# Conversation memory (in-memory for now)
# --------------------------------------------------
chat_history = []

# --------------------------------------------------
# Chat loop
# --------------------------------------------------
print("🤖 Customer Support Bot (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("AI: Thank you for contacting support. Have a great day!")
        break

    # Invoke model with history
    response = chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    print("AI:", response.content)

    # Save conversation
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
