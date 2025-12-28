from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=200,
    temperature=0.4,
)

chat_model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a Summary for the following poem -\n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('D:\GenerativeAI\langchain.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))
print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | chat_model | parser

result = chain.invoke({'poem': docs[0].page_content})

print(result)