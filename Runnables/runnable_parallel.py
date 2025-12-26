# we are trying to make chain with the help of runnable.
# here two chain run together.
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

prompt1 = PromptTemplate(
    template = 'Generate a Tweet About {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
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

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, chat_model, parser),
    'linkedin': RunnableSequence(prompt2, chat_model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})
print(result)