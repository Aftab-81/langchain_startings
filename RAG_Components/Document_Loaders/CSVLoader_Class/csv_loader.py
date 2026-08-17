from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = "Using this data: {data}.\n Answer this question: {question}",
    input_variables = ["question"]
)

parser = StrOutputParser()

loader = CSVLoader(file_path = "Document_Loaders\pokemon.csv")

docs = loader.load()

data_str = "\n\n".join(doc.page_content for doc in docs)

chain = template | model | parser

print(chain.invoke({
    "question": "Which pokemon is GOAT in this list?",
    "data": data_str
}))

