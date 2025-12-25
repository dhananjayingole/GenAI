from dotenv import load_dotenv
import os
# LangChain Models
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# LangChain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


# Load environment variables
load_dotenv()

# Tokens
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ---------------------------
# HuggingFace Zephyr Model
# ---------------------------
hf_llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=200,
    temperature=0.4,
    return_full_text=False,  # IMPORTANT: prevents [USER] leakage
)

model1 = ChatHuggingFace(llm=hf_llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
template='Classify the sentiment of the following feedback text into positive or negative \n{feedbck} \n {format_instruction}',
input_variables=['feedback'],
)

classifier_chain = prompt1 | model1 | parser
classifier_chain.invoke({'feedback': 'This is a terrible smartphone'})