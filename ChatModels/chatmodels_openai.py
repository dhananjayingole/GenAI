from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

Chatmodel = ChatOpenAI(model = 'gpt-4', temperature = 0.1, max_completion_tokens=300)
result = Chatmodel.invoke("What is the Capital of India")

print(result) # it will give output with multiple things as prompt_token etc.
print(result.content) #only string return

