from langchain_community.document_loaders import TextLoader

# Initialize TextLoader with raw string (r"...") to prevent escape character issues in Windows file paths
loader = TextLoader(r"Document_Loaders\sun.txt")

# TextLoader loads the entire text file into memory as a single Document object.
# Even though it is only 1 file, LangChain returns a List[Document] for standard interface consistency across all loaders.
#
# Internal structure of docs:
# docs = [
#     Document(
#         page_content="Golden sovereign of the day...", 
#         metadata={'source': 'Document_Loaders\\sun.txt'}
#     )
# ]

# Execute the synchronous load operation
docs = loader.load()

print("\n--- Document Load Results ---")

# Verify that load() returns a list container
print(f"Object Type: {type(docs)}")  # <class 'list'>

# Number of Document objects in the list (1 for a single text file)
print(f"Document Count: {len(docs)}")  # Output: 1

# Access the text payload of the first Document in the list
print("\nPage Content:")
print(docs[0].page_content)

# Access metadata (e.g., source file path) attached to the Document
print("\nMetadata:")
print(docs[0].metadata)

print(docs)