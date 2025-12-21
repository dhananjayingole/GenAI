# this model use to store text into vector.
# [1]* Embedding with one line or a Sentence.
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_query("Delhi is the Capital of India")

print(str(result))

# [2]* Embedding with docs or multiple one.
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the Capital of india",
    "Kolkata is the Capital of Bengal",
    "paris is the Capital of France."
]

result = embedding.embed_documents(documents)

print(str(result))
