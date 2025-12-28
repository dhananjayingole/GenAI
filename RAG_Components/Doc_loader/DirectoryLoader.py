# it is used to load multiple documents from the folder or directory.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs = loader.load()
docs = loader.lazy_load() # it return a generator., thry are not all loaded at once. they are fetched one at a time as needed.

print(len(docs))

print(docs[0].page_content)
