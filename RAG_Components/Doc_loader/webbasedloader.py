# used to load & extract text content from the web pages.
# it uses beautifuSoup under the hood to parse HTML and extract visible text.
 
from langchain_community.document_loaders import WebBaseLoader

url = ''
loader = WebBaseLoader(url)

docs = loader.load()
print(len(docs))
print(docs[0].page_content)