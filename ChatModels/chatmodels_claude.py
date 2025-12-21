from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

Chatmodel = ChatAnthropic(model = 'claude-3-5-sonnet-20241022', temp = 0.7)

result = Chatmodel.invoke("What is Capital of India")

print(result.content)

