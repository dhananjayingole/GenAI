# stroutparser is the simplest output parser in langchain.
# it is used to parse the output of a LLM and return it as a plain string.
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
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
# 1st prompt ->detailed prompt
template1 = PromptTemplate(
    template='Write a Detailed Report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> Summary
template2 = PromptTemplate(
    template='Write a 5 points summary on the following text./n {text}',
    input_variables=['text']
)

# prompt1 = template1.invoke({'topic':'black hole'})

# result1 = chat_model.invoke(prompt1)

# prompt2 = template2.invoke({'text': result1.content})
# result2 = chat_model.invoke(prompt2)

# print(result2.content)

# instaed of mutliple thing we will create a Chain for pipeline
parser = StrOutputParser()

chain = template1 | chat_model | parser | template2 | chat_model | parser
result = chain.invoke({'topic':'black hole'})
print(result)