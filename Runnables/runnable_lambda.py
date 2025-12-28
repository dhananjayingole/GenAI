# it is a runnable primitive that allows you to apply custom python fuctions within an AI pipeline.
# .it acts as a middleware b/n different AI componenets, enabling preprocessing, transformation, APi Calls, filtering and post-processing.
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough, RunnableLambda

load_dotenv()

# let make a function.
def word_count(text):
    count = len(text.split())
    return count

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

prompt1 = PromptTemplate(
    template = 'Generate a Tweet About {topic}',
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
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = print(final_chain.invoke({'topic':'AI'}))