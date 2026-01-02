import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# -------------------------
# Load environment
# -------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN")

# -------------------------
# Embeddings
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Chroma DB
# -------------------------
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# -------------------------
# LLM (✅ CORRECT)
# -------------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    max_new_tokens=128,
    temperature=0.5,
)

# -------------------------
# Prompt
# -------------------------
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

# -------------------------
# RAG Chain
# -------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

# -------------------------
# Query
# -------------------------
query = "What is ChromaDB?"
result = qa.invoke({"query": query})

print("\n🧠 Answer:\n")
print(result["result"])
