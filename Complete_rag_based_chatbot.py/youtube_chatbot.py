from dotenv import load_dotenv
import os

# Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Web loader & VectorStore (community versions)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

# Hugging Face models & embeddings
from langchain.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings

# RAG chain & prompts
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# -------------------------
# 1️⃣ Load Environment Variables
# -------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env")

# -------------------------
# 2️⃣ Load Web-based Documents
# -------------------------
# You can provide any URL(s) to scrape text content
urls = [
    "https://en.wikipedia.org/wiki/India",  # Example web page
    "https://en.wikipedia.org/wiki/New_Delhi"
]

all_docs = []
for url in urls:
    loader = WebBaseLoader(url)
    docs = loader.load()
    all_docs.extend(docs)

print(f"Loaded {len(all_docs)} documents from web.")

# -------------------------
# 3️⃣ Split Documents into Chunks
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # size of each chunk
    chunk_overlap=50  # overlap between chunks
)

docs_split = text_splitter.split_documents(all_docs)
print(f"Split into {len(docs_split)} chunks.")

# -------------------------
# 4️⃣ Create Embeddings
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# -------------------------
# 5️⃣ Create Vector Store
# -------------------------
vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embedding_model,
    persist_directory="chroma_db"  # persist for future use
)

# -------------------------
# 6️⃣ Setup Retriever
# -------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

# -------------------------
# 7️⃣ Setup LLM
# -------------------------
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=256,
    temperature=0.5,
)

chat_model = ChatHuggingFace(llm=llm)

# -------------------------
# 8️⃣ Setup RAG QA Chain
# -------------------------
# Prompt Template for RAG
prompt_template = """You are an assistant helping answer questions based on retrieved documents.
Use the context below to answer the question.
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",   # can also use 'map_reduce' for large docs
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# -------------------------
# 9️⃣ Chat Loop (Example)
# -------------------------
while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        break
    
    response = qa_chain.run(query)
    print(f"\nBot: {response}")
