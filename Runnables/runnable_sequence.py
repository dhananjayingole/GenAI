# we are trying to make chain with the help of runnable.
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

prompt = PromptTemplate(
    template = 'Write a Joke about {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=200,
    temperature=0.4,
)

chat_model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableSequence(prompt, chat_model, parser)

print(chain.invoke({'topic':'AI'}))
