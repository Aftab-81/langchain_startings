from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader(file_path = "Document_Loaders\sample_experiment.pdf", mode = "page")

docs = loader.load()

print(type(docs))
print(len(docs))

print("\nPage1:\n")
print(docs[0])

print("\nPage2:\n" )

print(docs[1])

print("\nPage3:\n")

print(docs[2])