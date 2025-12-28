from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='D:\GenerativeAI\SampleSuperstore.xlsx')
docs = loader.load()

print(len(docs))