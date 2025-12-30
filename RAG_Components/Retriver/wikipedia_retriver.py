# it retrives the queries the Wikipedia API to fetch relevant content for a given query.
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import WikipediaRetriever

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=200,
    temperature=0.4,
)

chat_model = ChatHuggingFace(llm=llm)

# initialise the retriver
retriever = WikipediaRetriever(top_k_results=2,lang="en")

# define the Query
query = "The geopolitical history of India and Pakistan from the perspective of a chinese"

# get relevant Wikipedia documents
docs = retriever.invoke(query)

# Print retrieved content
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")