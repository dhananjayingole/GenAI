# it read pdf files and read page by page of pdf file.
# there are mutliple Different types of pdfloader which are used for different type of pdf.

from langchain_community.document_loaders import PyPDFLoader

# Use raw string or forward slashes
loader = PyPDFLoader(r'D:\GenerativeAI\Operating_System_Placement_Notes.pdf')
# OR
# loader = PyPDFLoader('D:/GenerativeAI/Operating_System_Placement_Notes.pdf')

docs = loader.load()
print(f"Loaded {len(docs)} documents")

print(docs[0].page_content)
print(docs[1].metadata)