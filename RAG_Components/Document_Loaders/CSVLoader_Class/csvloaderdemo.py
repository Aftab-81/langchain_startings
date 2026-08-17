from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = "Document_Loaders\pokemon.csv")

docs = loader.load()

print(len(docs))

print(type(docs[0+1]))

print(len(docs[0].page_content))

for document in docs:
    print(document)
    print("*" * 20)