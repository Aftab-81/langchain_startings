from langchain_community.document_loaders import WebBaseLoader

url = "https://en.wikipedia.org/wiki/Battle_Through_the_Heavens"

loader = WebBaseLoader(web_path = url)

docs = loader.load()

print(docs)

print("*" * 190)

print("\n", len(docs))

print("Page Content:\n",docs[0].page_content)
print("Metadata:\n", docs[0].metadata)