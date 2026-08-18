
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 370,
    chunk_overlap = 10
)

text = """
    # LangChain Text Splitting

LangChain is a framework for building applications with Large Language Models.
Text splitters divide large documents into smaller chunks.
Chunk size controls the maximum size of each chunk.
Chunk overlap repeats some content between consecutive chunks.
Overlap helps preserve context between chunks.
These chunks can later be used for RAG and retrieval.
"""

chunks = splitter.split_text(text = text)

print(len(chunks))
print(chunks)



