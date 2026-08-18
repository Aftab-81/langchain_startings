from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = CharacterTextSplitter(chunk_size = 100, chunk_overlap = 0, separator = "" )

loader = PyPDFLoader(file_path = r"RAG_Components\Text_Splitters\Length_Based_Text_Splitter\book\ML Book.pdf")

docs = loader.load()

chunks = splitter.split_text(text = "\n\n".join(document.page_content for document in docs))

print(chunks)