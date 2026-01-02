import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.schema import Document

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Documents
docs = [
    Document(page_content="Photosynthesis is the process by which green plants convert sunlight into chemical energy."),
    Document(page_content="Chlorophyll absorbs sunlight during photosynthesis in plant cells."),
    Document(page_content="Basketball was invented by Dr. James Naismith."),
    Document(page_content="Photosynthesis does not occur in animal cells."),
]

# Embeddings + Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(docs, embeddings)

# LLM (Hugging Face)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.3,
    max_new_tokens=200,
)

# Contextual Compression Retriever
compressor = LLMChainExtractor.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    base_compressor=compressor,
)

# Query
results = retriever.invoke("What is photosynthesis?")

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
