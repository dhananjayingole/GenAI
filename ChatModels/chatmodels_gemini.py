from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize Gemini Chat Model
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

# Chat messages (role-based)
messages = [
    SystemMessage(content="You are a helpful AI tutor."),
    HumanMessage(content="Explain LangChain in simple terms.")
]

# Invoke Chat Model
response = chat.invoke(messages)

print("\n--- Gemini ChatModel Response ---\n")
print(response.content)
