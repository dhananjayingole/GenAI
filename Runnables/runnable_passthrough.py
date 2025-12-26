# this Runnable passthrough is a sp runnable primitive that simply return the input as output without modifiying it.
# let first we create a Joke then will call parallel_chain in which 
# ->at one side it will call Runnable_passthrough which will only gives joke as it get joke in inout.
# -> at other side it will call other Runnable_parallel which will give joke explanations.
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

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

joke_gen_chain = RunnableSequence(prompt1, chat_model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, chat_model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))