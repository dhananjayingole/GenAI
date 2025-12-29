from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. Initialize the Open Source Embedding Model
# This will now work once you've run 'pip install sentence-transformers'
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# 2. Create LangChain documents using the Document class (not TextLoader)
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
]

# 3. Initialize ChromaDB
vector_store = Chroma(
    embedding_function=embedding_function,
    persist_directory='my_chroma_db_hf',
    collection_name='ipl_players_hf'
)

# 4. Add documents to the store
print("Adding documents to Vector Store...")
vector_store.add_documents(docs)

# 5. Perform Similarity Search
query = 'Who among these are a bowler?'
print(f"\nQuerying: {query}")

results = vector_store.similarity_search(query, k=2)

# 6. Display Results
print("\n--- Search Results ---")
for res in results:
    print(f"Player Info: {res.page_content}")
    print(f"Metadata: {res.metadata}\n")