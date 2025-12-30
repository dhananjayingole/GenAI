from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=200,
    temperature=0.4,
)

docs = [
    Document(page_content="Regular walking boosts heart health and energy."),
    Document(page_content="Deep sleep improves emotional and physical balance."),
    Document(page_content="Mindfulness lowers cortisol and improves focus."),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(k=3),
    llm=llm
)

query = "How can I improve energy and balance?"

for doc in retriever.invoke(query):
    print("-", doc.page_content)
